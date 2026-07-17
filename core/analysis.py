from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests

from core.config import AppConfig, ConfigStore
from core.errors import AnalysisError, InferenceError, ServiceError
from core.inference.llm_client import LLMClient
from core.inference.sam_client import SAMClient
from core.pipeline.priority_map_adapter import PipelineResult, PriorityMapAdapter
from core.services.controller import ServiceController
from core.services.instruction_client import InstructionClient
from core.services.llm_settings import (
    REMOTE_LLM_KEYS,
    generation_settings,
    resolve_inference_bind_host,
    validate_llm_settings,
)
from core.services.specs import ServiceEndpoint, ServiceSpec

AnalysisState = Literal["idle", "starting", "running", "cancelling", "cancelled", "completed", "failed"]


@dataclass(slots=True)
class AnalysisStatus:
    state: AnalysisState = "idle"
    message: str = ""
    input_path: str = ""
    output_directory: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    frames_processed: int = 0
    error: str | None = None
    output_paths: list[str] = field(default_factory=list)
    run_id: str = ""
    stage: str = ""
    stream_url: str | None = None
    artifacts_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnalysisCoordinator:
    """Run one client-side Priority Map analysis in a managed thread."""

    def __init__(
        self,
        config_store: ConfigStore | None = None,
        controller: ServiceController | None = None,
        adapter: PriorityMapAdapter | None = None,
    ) -> None:
        self.config_store = config_store
        self.controller = controller or ServiceController()
        self.adapter = adapter or PriorityMapAdapter()
        self._lock = threading.RLock()
        self._preview_condition = threading.Condition(self._lock)
        self._status = AnalysisStatus()
        self._thread: threading.Thread | None = None
        self._config: AppConfig | None = None
        self._endpoints: dict[str, ServiceEndpoint] = {}
        self._cancel_event = threading.Event()
        self._latest_preview: bytes | None = None
        self._preview_version = 0

    def status(self) -> AnalysisStatus:
        with self._lock:
            return AnalysisStatus(**self._status.to_dict())

    def is_active(self) -> bool:
        return self.status().state in ("starting", "running", "cancelling")

    def assert_configuration_mutable(self) -> None:
        if self.is_active():
            raise AnalysisError("Service and pipeline configuration cannot change during an active analysis.")

    def preview(self, version: int = 0, timeout: float = 0.5) -> tuple[bytes | None, int, AnalysisState]:
        with self._preview_condition:
            if self._preview_version <= version and self._status.state not in ("completed", "cancelled", "failed"):
                self._preview_condition.wait(timeout)
            return self._latest_preview, self._preview_version, self._status.state

    def start(self, input_path: str | Path, config: AppConfig | None = None) -> AnalysisStatus:
        with self._lock:
            if self.is_active():
                raise AnalysisError("An analysis is already running.")
            effective_config = config or (self.config_store.load() if self.config_store else AppConfig())
            input_value = str(Path(input_path).expanduser())
            output_directory = self._allocate_output_directory(effective_config)
            run_id = output_directory.name
            self._cancel_event = threading.Event()
            self._latest_preview = None
            self._preview_version = 0
            self._status = self._new_status(input_value, output_directory, run_id)
            self._config = effective_config
            self._thread = threading.Thread(
                target=self._run,
                args=(input_value, effective_config, output_directory, run_id),
                name="arcadia-analysis",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def run_sync(self, input_path: str | Path, config: AppConfig | None = None) -> AnalysisStatus:
        with self._lock:
            if self.is_active():
                raise AnalysisError("An analysis is already running.")
            effective_config = config or (self.config_store.load() if self.config_store else AppConfig())
            output_directory = self._allocate_output_directory(effective_config)
            run_id = output_directory.name
            self._cancel_event = threading.Event()
            self._latest_preview = None
            self._preview_version = 0
            self._status = self._new_status(str(input_path), output_directory, run_id)
            self._config = effective_config
        self._run(str(input_path), effective_config, output_directory, run_id)
        return self.status()

    def cancel_after_current_frame(self) -> AnalysisStatus:
        with self._preview_condition:
            if self._status.state not in ("starting", "running", "cancelling"):
                raise AnalysisError("No active analysis can be cancelled.")
            if self._status.state != "cancelling":
                stage = self._status.stage
                if stage == "preparing_input":
                    message = "Cancelling after current preparation step"
                elif stage in ("validating", "starting_llm", "starting_sam3"):
                    message = "Cancelling after current startup step"
                else:
                    message = "Cancelling after current frame"
                self._cancel_event.set()
                self._status.state = "cancelling"
                self._status.stage = "cancelling"
                self._status.message = message
                self._preview_condition.notify_all()
            return self.status()

    @staticmethod
    def _new_status(input_path: str, output_directory: Path, run_id: str) -> AnalysisStatus:
        prefix = f"/client/priority-map/runs/{run_id}"
        return AnalysisStatus(
            state="starting",
            message="Preparing services",
            input_path=input_path,
            output_directory=str(output_directory),
            started_at=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
            stage="validating",
            stream_url=f"{prefix}/stream/",
            artifacts_url=f"{prefix}/artifacts/",
        )

    @staticmethod
    def _allocate_output_directory(config: AppConfig) -> Path:
        output_root = Path(config.priority_map.output.root)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for suffix in range(1000):
            name = timestamp if suffix == 0 else f"{timestamp}-{suffix:03d}"
            candidate = output_root / name
            try:
                candidate.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            return candidate
        raise AnalysisError("Could not allocate a unique analysis output directory.")

    def _run(self, input_path: str, config: AppConfig, output_directory: Path, run_id: str) -> None:
        log_path = output_directory / "analysis.log"
        with log_path.open("a", encoding="utf-8") as log:
            try:
                self._set_status(stage="validating", message="Validating run")
                if self._cancelled_checkpoint(output_directory):
                    return
                self._write_effective_settings(config, input_path, output_directory, run_id)
                if self._cancelled_checkpoint(output_directory):
                    return

                pm_config = config.priority_map
                graph_agent_enabled = bool(pm_config.pipeline.graph_agent)

                # Determine which services to start
                need_logical = graph_agent_enabled
                visual_mode = pm_config.visual_llm_mode

                llm_endpoint: ServiceEndpoint | None = None
                visual_endpoint: ServiceEndpoint | None = None

                if visual_mode == "same_as_logical":
                    # Start Logical LLM once for both text and vision
                    self._set_status(stage="starting_llm", message="Starting Logical LLM (shared mode)")
                    llm_endpoint = self._ensure_service("llm", config, output_directory)
                    visual_endpoint = llm_endpoint
                    if self._cancelled_checkpoint(output_directory):
                        return
                else:
                    # Separate mode: start Visual for scene understanding
                    self._set_status(stage="starting_visual_llm", message="Starting Visual LLM service")
                    visual_endpoint = self._ensure_service("visual_llm", config, output_directory)
                    if self._cancelled_checkpoint(output_directory):
                        return

                    # Start Logical separately only if Graph Agent needs it
                    if need_logical:
                        self._set_status(stage="starting_llm", message="Starting Logical LLM service")
                        llm_endpoint = self._ensure_service("llm", config, output_directory)
                        if self._cancelled_checkpoint(output_directory):
                            return

                self._set_status(stage="starting_sam3", message="Starting SAM3 service")
                if self._cancelled_checkpoint(output_directory):
                    return
                sam_endpoint = self._ensure_service("sam3", config, output_directory)
                if self._cancelled_checkpoint(output_directory):
                    return

                self._set_status(state="running", stage="preparing_input", message="Preparing input")
                settings = config.priority_map.pipeline.to_dict()

                # Use the correct service's generation settings for each role
                logical_service = config.priority_map.services.get("llm")
                if visual_mode == "separate":
                    visual_service = config.priority_map.services.get("visual_llm")
                    settings["llm_generation"] = generation_settings(
                        logical_service.settings if logical_service else {}
                    )
                    settings["visual_llm_generation"] = generation_settings(
                        visual_service.settings if visual_service else {}
                    )
                else:
                    # Shared mode: both use Logical settings
                    gen = generation_settings(logical_service.settings if logical_service else {})
                    settings["llm_generation"] = gen
                    settings["visual_llm_generation"] = gen

                # Pass model aliases for the adapter
                if logical_service:
                    settings["logical_model_alias"] = logical_service.settings.get("model_alias", "")
                if visual_mode == "separate":
                    visual_service = config.priority_map.services.get("visual_llm")
                    if visual_service:
                        settings["visual_model_alias"] = visual_service.settings.get("model_alias", "visual-model")
                    else:
                        settings["visual_model_alias"] = "visual-model"
                else:
                    # Shared mode: Visual uses the same Logical alias
                    if logical_service:
                        settings["visual_model_alias"] = logical_service.settings.get("model_alias", "")

                settings["output_directory"] = str(output_directory)
                settings["input_path"] = input_path
                self._set_status(stage="priority_map", message="Running Priority Map")
                if self._cancelled_checkpoint(output_directory):
                    return

                # Ensure llm_endpoint is set for graph agent when needed
                if need_logical and llm_endpoint is None:
                    self._set_status(stage="starting_llm", message="Starting Logical LLM for Graph Agent")
                    llm_endpoint = self._ensure_service("llm", config, output_directory)

                llm_client = LLMClient(llm_endpoint or visual_endpoint)
                visual_llm_client = LLMClient(visual_endpoint)
                result = self._run_adapter(
                    input_path,
                    output_directory,
                    llm_endpoint or visual_endpoint,
                    visual_endpoint,
                    sam_endpoint,
                    settings,
                    llm_client,
                    visual_llm_client,
                )
                self._set_status(stage="finalizing", message="Finalizing output")
                if self._cancel_event.is_set():
                    self._finish_cancelled(output_directory, result.frames_processed)
                else:
                    self._finish_completed(result, log)
            except (InferenceError, ServiceError) as exc:
                if self._cancel_event.is_set():
                    self._log(log, f"analysis cancelled after service error: {exc}")
                    self._finish_cancelled(output_directory)
                else:
                    self._log(log, f"analysis failed: {exc}")
                    self._finish_failed(str(exc))
            except Exception as exc:
                if self._cancel_event.is_set():
                    self._log(log, f"analysis cancelled: {exc}")
                    self._finish_cancelled(output_directory)
                else:
                    self._log(log, f"analysis failed: {exc}")
                    self._finish_failed(str(exc))

    def _ensure_service(self, name: str, config: AppConfig, output_directory: Path) -> ServiceEndpoint:
        configured = config.priority_map.services.get(name)
        if configured is None:
            raise AnalysisError(f"Required service configuration is missing: {name}")
        node = config.nodes.get(configured.node)
        if node is None:
            raise AnalysisError(f"Service {name} references unknown node {configured.node!r}")
        if node.mode == "local":
            endpoint = self.controller.start(configured.spec, cancel_event=self._cancel_event)
        else:
            if node.instruction_port is None:
                raise AnalysisError(f"Remote node {configured.node!r} has no instruction port")
            if self._cancelled_checkpoint(output_directory):
                raise ServiceError("Analysis cancelled before remote service startup.")
            spec = configured.spec
            if name in ("llm", "visual_llm"):
                settings = {key: value for key, value in configured.settings.items() if key in REMOTE_LLM_KEYS}
                settings = validate_llm_settings(settings, remote=True)
                settings["bind_host"] = resolve_inference_bind_host(configured.node, config.nodes, None)
                spec = ServiceSpec(service_type=name, port=configured.port, settings=settings)
            endpoint = InstructionClient(node.host, node.instruction_port).start_service(spec)
            if self._cancelled_checkpoint(output_directory):
                raise ServiceError("Analysis cancelled after remote service startup.")
            self._wait_ready(
                endpoint,
                float(configured.settings.get("startup_timeout", 600)),
                cancel_event=self._cancel_event,
            )
        self._endpoints[name] = endpoint
        return endpoint

    def _run_adapter(
        self,
        input_path: str,
        output_directory: Path,
        llm_endpoint: ServiceEndpoint,
        visual_endpoint: ServiceEndpoint,
        sam_endpoint: ServiceEndpoint,
        settings: dict[str, Any],
        llm_client: LLMClient,
        visual_llm_client: LLMClient,
    ) -> PipelineResult:
        attempts = 0
        while True:
            try:
                return self.adapter.run(
                    input_path=input_path,
                    output_directory=str(output_directory),
                    llm_client=llm_client,
                    visual_llm_client=visual_llm_client,
                    sam_client=SAMClient(sam_endpoint),
                    pipeline_settings=settings,
                    progress_callback=self._progress,
                    cancel_event=self._cancel_event,
                    preview_callback=self._publish_preview,
                )
            except (InferenceError, ServiceError) as exc:
                if self._cancel_event.is_set() or attempts >= 1:
                    raise
                attempts += 1
                service_type = getattr(exc, "service_type", None) if isinstance(exc, InferenceError) else None
                # Determine which service to restart based on the error
                if service_type and service_type not in ("llm", "visual_llm", "sam3"):
                    service_type = None
                if service_type is None:
                    # Fallback: restart the service that corresponds to the endpoint
                    service_names = ["llm", "sam3"]
                    if visual_endpoint.service_type == "visual_llm":
                        service_names.append("visual_llm")
                else:
                    service_names = [service_type]
                for service_name in service_names:
                    if self._cancelled_checkpoint(output_directory):
                        raise ServiceError("Analysis cancelled before service restart.") from exc
                    self._endpoints[service_name] = self._ensure_service(
                        service_name, self._config or AppConfig(), output_directory
                    )
                # Update clients and endpoints
                llm_endpoint = self._endpoints.get("llm", llm_endpoint)
                sam_endpoint = self._endpoints.get("sam3", sam_endpoint)
                visual_endpoint = self._endpoints.get("visual_llm", visual_endpoint)
                if llm_endpoint.service_type in ("llm", "visual_llm"):
                    llm_client = LLMClient(llm_endpoint)
                visual_llm_client = LLMClient(visual_endpoint)

    @staticmethod
    def _wait_ready(endpoint: ServiceEndpoint, timeout: float, *, cancel_event: threading.Event | None = None) -> None:
        path = "/v1/models" if endpoint.service_type == "llm" else "/health"
        deadline = time.monotonic() + max(1.0, timeout)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise ServiceError(f"{endpoint.service_type} startup cancelled.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceError(
                    f"Timed out waiting for {endpoint.service_type} readiness at {endpoint.base_url}{path}"
                )
            try:
                response = requests.get(f"{endpoint.base_url}{path}", timeout=min(2.0, max(0.1, remaining)))
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ServiceError(
                    f"Timed out waiting for {endpoint.service_type} readiness at {endpoint.base_url}{path}"
                )
            if cancel_event is not None:
                if cancel_event.wait(min(0.1, remaining)):
                    raise ServiceError(f"{endpoint.service_type} startup cancelled.")
            else:
                time.sleep(min(0.1, remaining))

    def _write_effective_settings(
        self, config: AppConfig, input_path: str, output_directory: Path, run_id: str
    ) -> None:
        payload = config.priority_map.to_dict()
        payload.update(
            {
                "run_id": run_id,
                "input_path": input_path,
                "nodes": {name: node.to_dict() for name, node in config.nodes.items()},
                "effective_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        (output_directory / "effective_settings.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def _progress(self, progress: dict[str, Any]) -> None:
        with self._lock:
            self._status.frames_processed = int(progress.get("frames_processed", self._status.frames_processed) or 0)
            if self._status.state != "cancelling":
                frame_index = progress.get("frame_index")
                self._status.message = (
                    f"Processed frame {frame_index}" if frame_index is not None else "Processing frames"
                )

    def _publish_preview(self, jpeg: bytes) -> None:
        with self._preview_condition:
            self._latest_preview = jpeg
            self._preview_version += 1
            self._preview_condition.notify_all()

    def _set_status(self, **changes: Any) -> None:
        with self._preview_condition:
            for key, value in changes.items():
                setattr(self._status, key, value)
            self._preview_condition.notify_all()

    def _cancelled_checkpoint(self, output_directory: Path) -> bool:
        if not self._cancel_event.is_set():
            return False
        self._finish_cancelled(output_directory)
        return True

    def _finish_completed(self, result: PipelineResult, log: Any) -> None:
        self._log(log, "analysis completed")
        with self._preview_condition:
            if self._cancel_event.is_set():
                self._finish_cancelled(Path(result.output_directory), result.frames_processed)
                return
            self._status.state = "completed"
            self._status.stage = "finalizing"
            self._status.message = "Analysis completed"
            self._status.finished_at = datetime.now(timezone.utc).isoformat()
            self._status.frames_processed = result.frames_processed
            self._status.output_paths = self._files_under(Path(result.output_directory))
            self._preview_condition.notify_all()

    def _finish_cancelled(self, output_directory: Path, frames_processed: int | None = None) -> None:
        with self._preview_condition:
            self._status.state = "cancelled"
            self._status.stage = "finalizing"
            self._status.message = "Analysis cancelled"
            self._status.finished_at = datetime.now(timezone.utc).isoformat()
            if frames_processed is not None:
                self._status.frames_processed = frames_processed
            self._status.output_paths = self._files_under(output_directory)
            self._preview_condition.notify_all()

    def _finish_failed(self, error: str) -> None:
        with self._preview_condition:
            self._status.state = "failed"
            self._status.stage = "finalizing"
            self._status.message = "Analysis failed"
            self._status.error = error
            self._status.finished_at = datetime.now(timezone.utc).isoformat()
            self._status.output_paths = self._files_under(Path(self._status.output_directory))
            self._preview_condition.notify_all()

    @staticmethod
    def _files_under(directory: Path) -> list[str]:
        return sorted(str(path) for path in directory.rglob("*") if path.is_file()) if directory.exists() else []

    @staticmethod
    def _log(handle: Any, message: str) -> None:
        handle.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
        handle.flush()
