from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from core.config import AppConfig, ConfiguredService
from core.errors import AnalysisError
from core.inference import LLMClient, SAMClient
from core.pipeline import PriorityMapAdapter
from core.services.controller import ServiceController
from core.services.instruction_client import InstructionClient
from core.services.llm_settings import generation_settings, validate_llm_settings
from core.services.specs import ServiceEndpoint, ServiceSpec, ServiceType


@dataclass(frozen=True, slots=True)
class AnalysisStatus:
    state: str = "idle"
    run_id: str | None = None
    input_path: str | None = None
    output_directory: str | None = None
    frames_processed: int = 0
    image_name: str | None = None
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "run_id": self.run_id,
            "input_path": self.input_path,
            "output_directory": self.output_directory,
            "frames_processed": self.frames_processed,
            "image_name": self.image_name,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "artifacts": list(self.artifacts),
        }


class AnalysisCoordinator:
    def __init__(self, controller: ServiceController, adapter: PriorityMapAdapter | None = None) -> None:
        self.controller = controller
        self.adapter = adapter or PriorityMapAdapter()
        self._lock = threading.RLock()
        self._preview_condition = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._status = AnalysisStatus()
        self._preview_bytes: bytes | None = None
        self._preview_version = 0

    def status(self) -> AnalysisStatus:
        with self._lock:
            return AnalysisStatus(**self._status.to_dict())

    def is_active(self) -> bool:
        return self.status().state in {"starting", "running", "cancelling"}

    def assert_configuration_mutable(self) -> None:
        if self.is_active():
            raise AnalysisError("Model and pipeline settings cannot be changed while an analysis is active.")

    def start(self, input_path: Path, config: AppConfig) -> AnalysisStatus:
        with self._lock:
            if self.is_active():
                raise AnalysisError("An analysis is already active.")
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
            output_root = config.priority_map.output.root
            output_directory = Path(output_root) / run_id
            output_directory.mkdir(parents=True, exist_ok=False)
            snapshot = AppConfig.from_dict(config.to_dict())
            self._cancel = threading.Event()
            self._preview_bytes = None
            self._preview_version = 0
            self._status = AnalysisStatus(
                state="starting",
                run_id=run_id,
                input_path=str(input_path),
                output_directory=str(output_directory),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(input_path, output_directory, snapshot),
                name=f"arcadia-analysis-{run_id}",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def cancel_after_current_frame(self) -> AnalysisStatus:
        with self._lock:
            if not self.is_active():
                raise AnalysisError("No analysis is active.")
            self._cancel.set()
            self._replace_status(state="cancelling")
            return self.status()

    def preview(self, version: int, timeout: float = 1.0) -> tuple[bytes | None, int, str]:
        with self._preview_condition:
            if self._preview_version <= version and self.is_active():
                self._preview_condition.wait(timeout=timeout)
            return self._preview_bytes, self._preview_version, self._status.state

    def _replace_status(self, **updates: Any) -> None:
        payload = self._status.to_dict()
        payload.update(updates)
        self._status = AnalysisStatus(**payload)

    def _set_preview(self, jpeg: bytes) -> None:
        with self._preview_condition:
            self._preview_bytes = jpeg
            self._preview_version += 1
            self._preview_condition.notify_all()

    def _progress(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._replace_status(
                state="cancelling" if self._cancel.is_set() else "running",
                frames_processed=int(payload.get("frames_processed", self._status.frames_processed) or 0),
                image_name=payload.get("image_name") or self._status.image_name,
            )

    def _configured(self, config: AppConfig, role: str) -> tuple[str, ConfiguredService]:
        backing = role
        if role == "visual_llm" and config.priority_map.visual_llm_mode == "same_as_logical":
            backing = "llm"
        configured = config.priority_map.services.get(backing)
        if configured is None:
            raise AnalysisError(f"No saved configuration exists for {backing}.")
        return backing, configured

    def _start_service(self, config: AppConfig, backing_role: str, configured: ConfiguredService) -> ServiceEndpoint:
        node = config.nodes.get(configured.node)
        if node is None:
            raise AnalysisError(f"{backing_role} references unknown node {configured.node!r}.")
        spec = configured.spec
        if backing_role in ("llm", "visual_llm"):
            spec = ServiceSpec(
                cast(ServiceType, backing_role),
                configured.port,
                validate_llm_settings(configured.settings, remote=node.mode == "remote"),
            )
        if node.mode == "local":
            if self.controller.matches(spec):
                return self.controller.endpoint_for(spec.port)
            return self.controller.start(spec, cancel_event=self._cancel)
        if node.instruction_port is None:
            raise AnalysisError(f"Remote node {configured.node!r} has no instruction port.")
        return InstructionClient(node.host, node.instruction_port).ensure_service(spec)

    def _run(self, input_path: Path, output_directory: Path, config: AppConfig) -> None:
        log_path = output_directory / "analysis.log"
        try:
            (output_directory / "effective_settings.json").write_text(
                json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self._lock:
                self._replace_status(state="running")

            logical_role, logical_config = self._configured(config, "llm")
            visual_role, visual_config = self._configured(config, "visual_llm")
            sam_role, sam_config = self._configured(config, "sam3")
            logical_endpoint = self._start_service(config, logical_role, logical_config)
            visual_endpoint = (
                logical_endpoint
                if visual_config is logical_config
                else self._start_service(config, visual_role, visual_config)
            )
            sam_endpoint = self._start_service(config, sam_role, sam_config)

            logical_generation = generation_settings(logical_config.settings)
            visual_generation = generation_settings(visual_config.settings)
            logical_client = LLMClient(
                logical_endpoint,
                role_defaults=logical_generation,
            )
            visual_client = LLMClient(
                visual_endpoint,
                role_defaults=visual_generation,
            )
            sam_client = SAMClient(sam_endpoint)
            pipeline_settings = config.priority_map.pipeline.to_dict()
            pipeline_settings.update(
                {
                    "llm_generation": logical_generation,
                    "visual_llm_generation": visual_generation,
                }
            )
            result = self.adapter.run(
                input_path=str(input_path),
                output_directory=str(output_directory),
                llm_client=logical_client,
                visual_llm_client=visual_client,
                sam_client=sam_client,
                pipeline_settings=pipeline_settings,
                progress_callback=self._progress,
                cancel_event=self._cancel,
                preview_callback=self._set_preview,
            )
            artifacts = [str(Path(path).relative_to(output_directory)) for path in result.output_paths]
            state = "cancelled" if self._cancel.is_set() else "completed"
            with self._preview_condition:
                self._replace_status(
                    state=state,
                    frames_processed=result.frames_processed,
                    artifacts=artifacts,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                self._preview_condition.notify_all()
            log_path.write_text(f"Analysis {state}. Frames processed: {result.frames_processed}\n", encoding="utf-8")
        except Exception as exc:
            try:
                log_path.write_text(f"Analysis failed: {type(exc).__name__}: {exc}\n", encoding="utf-8")
            except OSError:
                pass
            with self._preview_condition:
                self._replace_status(
                    state="failed",
                    error=str(exc),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                self._preview_condition.notify_all()
