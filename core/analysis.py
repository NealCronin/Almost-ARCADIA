from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from core.config import AppConfig, ConfigStore
from core.errors import AnalysisError, InferenceError, ServiceError
from core.inference.llm_client import LLMClient
from core.inference.sam_client import SAMClient
from core.pipeline.priority_map_adapter import PipelineResult, PriorityMapAdapter
from core.services.controller import ServiceController
from core.services.instruction_client import InstructionClient
from core.services.specs import ServiceEndpoint

AnalysisState = Literal["idle", "starting", "running", "completed", "failed"]


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
        self._status = AnalysisStatus()
        self._thread: threading.Thread | None = None
        self._config: AppConfig | None = None
        self._endpoints: dict[str, ServiceEndpoint] = {}

    def status(self) -> AnalysisStatus:
        with self._lock:
            current = self._status
            return AnalysisStatus(**current.to_dict())

    def is_active(self) -> bool:
        return self.status().state in ("starting", "running")

    def assert_configuration_mutable(self) -> None:
        if self.is_active():
            raise AnalysisError("Service and pipeline configuration cannot change during an active analysis.")

    def start(self, input_path: str | Path, config: AppConfig | None = None) -> AnalysisStatus:
        with self._lock:
            if self._status.state in ("starting", "running"):
                raise AnalysisError("An analysis is already running.")
            effective_config = config or (self.config_store.load() if self.config_store else AppConfig())
            input_value = str(Path(input_path).expanduser())
            output_directory = self._allocate_output_directory(effective_config)
            self._status = AnalysisStatus(
                state="starting",
                message="Preparing services",
                input_path=input_value,
                output_directory=str(output_directory),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self._config = effective_config
            self._thread = threading.Thread(
                target=self._run,
                args=(input_value, effective_config, output_directory),
                name="arcadia-analysis",
                daemon=True,
            )
            self._thread.start()
            return self.status()

    def run_sync(self, input_path: str | Path, config: AppConfig | None = None) -> AnalysisStatus:
        """Synchronous entry point used by integration tests and CLI smoke runs."""
        with self._lock:
            if self.is_active():
                raise AnalysisError("An analysis is already running.")
            effective_config = config or (self.config_store.load() if self.config_store else AppConfig())
            output_directory = self._allocate_output_directory(effective_config)
            self._status = AnalysisStatus(
                state="starting",
                input_path=str(input_path),
                output_directory=str(output_directory),
                started_at=datetime.now(timezone.utc).isoformat(),
                message="Preparing services",
            )
        self._run(str(input_path), effective_config, output_directory)
        return self.status()

    @staticmethod
    def _allocate_output_directory(config: AppConfig) -> Path:
        output_root = Path(config.output_root)
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

    def _run(self, input_path: str, config: AppConfig, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        log_path = output_directory / "analysis.log"
        with log_path.open("a", encoding="utf-8") as log:
            try:
                self._write_effective_settings(config, input_path, output_directory)
                self._log(log, "analysis starting")
                self._set_status(state="running", message="Starting inference services")
                llm_endpoint = self._ensure_service("llm", config)
                sam_endpoint = self._ensure_service("sam3", config)
                self._set_status(message="Running Priority Map")
                llm_client = LLMClient(llm_endpoint)
                sam_client = SAMClient(sam_endpoint)
                settings = config.pipeline.to_dict()
                settings["output_directory"] = str(output_directory)
                settings["input_path"] = input_path
                attempts = 0
                while True:
                    try:
                        result = self.adapter.run(
                            input_path=input_path,
                            output_directory=str(output_directory),
                            llm_client=llm_client,
                            sam_client=sam_client,
                            pipeline_settings=settings,
                            progress_callback=lambda progress: self._progress(progress),
                        )
                        break
                    except (InferenceError, ServiceError) as exc:
                        if attempts >= 1:
                            raise AnalysisError(f"Service failure after one restart/retry: {exc}") from exc
                        attempts += 1
                        service_names = (
                            (exc.service_type,)
                            if isinstance(exc, InferenceError) and exc.service_type
                            else (
                                "llm",
                                "sam3",
                            )
                        )
                        self._log(log, f"service failure; restarting {', '.join(service_names)} once: {exc}")
                        for service_name in service_names:
                            self._endpoints[service_name] = self._ensure_service(service_name, config, force=True)
                        llm_client = LLMClient(self._endpoints["llm"])
                        sam_client = SAMClient(self._endpoints["sam3"])
                self._finish_completed(result, log)
            except Exception as exc:
                self._log(log, f"analysis failed: {exc}")
                self._finish_failed(str(exc))

    def _ensure_service(self, name: str, config: AppConfig, force: bool = False) -> ServiceEndpoint:
        configured = config.services.get(name)
        if configured is None:
            raise AnalysisError(f"Required service configuration is missing: {name}")
        node = config.nodes.get(configured.node)
        if node is None:
            raise AnalysisError(f"Service {name} references unknown node {configured.node!r}")
        spec = configured.spec
        if node.mode == "local":
            if force or not self.controller.is_running(spec.port):
                endpoint = self.controller.start(spec)
            else:
                endpoint = ServiceEndpoint(node.host, spec.port, spec.service_type)
        else:
            if node.instruction_port is None:
                raise AnalysisError(f"Remote node {configured.node!r} has no instruction port")
            client = InstructionClient(node.host, node.instruction_port)
            if force or not any(status.port == spec.port and status.running for status in client.list_services()):
                endpoint = client.start_service(spec)
            else:
                endpoint = ServiceEndpoint(node.host, spec.port, spec.service_type)
        self._endpoints[name] = endpoint
        return endpoint

    def _write_effective_settings(self, config: AppConfig, input_path: str, output_directory: Path) -> None:
        payload = config.to_dict()
        payload["input_path"] = input_path
        payload["effective_at"] = datetime.now(timezone.utc).isoformat()
        (output_directory / "effective_settings.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
        )

    def _progress(self, progress: dict[str, Any]) -> None:
        with self._lock:
            self._status.frames_processed = int(progress.get("frames_processed", self._status.frames_processed) or 0)
            frame_index = progress.get("frame_index")
            self._status.message = f"Processed frame {frame_index}" if frame_index is not None else "Processing frames"

    def _set_status(self, **changes: Any) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._status, key, value)

    def _finish_completed(self, result: PipelineResult, log: Any) -> None:
        self._log(log, "analysis completed")
        with self._lock:
            self._status.state = "completed"
            self._status.message = "Analysis completed"
            self._status.finished_at = datetime.now(timezone.utc).isoformat()
            self._status.frames_processed = result.frames_processed
            self._status.output_paths = list(result.output_paths)

    def _finish_failed(self, error: str) -> None:
        with self._lock:
            self._status.state = "failed"
            self._status.message = "Analysis failed"
            self._status.error = error
            self._status.finished_at = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _log(handle: Any, message: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        handle.write(f"{timestamp} {message}\n")
        handle.flush()
