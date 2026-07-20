from __future__ import annotations

import atexit
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from core.errors import ServiceError, ServiceNotRunningError, ServiceStartupError
from core.services.llm_runtime import LLMRuntime
from core.services.specs import ServiceEndpoint, ServiceSpec, ServiceStatus


@dataclass(slots=True)
class RunningService:
    spec: ServiceSpec
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: IO[str]
    endpoint: ServiceEndpoint


class ServiceController:
    """Own only child processes launched by this controller instance."""

    def __init__(
        self,
        public_host: str = "127.0.0.1",
        log_dir: str | Path = "logs",
        *,
        startup_timeout: float = 600.0,
        allow_test_commands: bool = False,
    ) -> None:
        self.public_host = public_host
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = startup_timeout
        self.allow_test_commands = allow_test_commands
        self._services: dict[int, RunningService] = {}
        self._lock = threading.RLock()
        atexit.register(self.stop_all)

    def _runtime_for(self, spec: ServiceSpec) -> Any:
        if spec.service_type in ("llm", "visual_llm"):
            return LLMRuntime
        if spec.service_type == "sam3":
            from core.services.sam_runtime import SAMRuntime

            return SAMRuntime
        raise ValueError(f"Unsupported service type: {spec.service_type}")

    @staticmethod
    def _startup_log_tail(log_path: Path, limit: int = 40) -> str:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        tail = "\n".join(lines[-limit:]).strip()
        return f"\n\nservice log tail:\n{tail}" if tail else ""

    def start(self, spec: ServiceSpec, *, cancel_event: threading.Event | None = None) -> ServiceEndpoint:
        with self._lock:
            if cancel_event is not None and cancel_event.is_set():
                raise ServiceStartupError(f"{spec.service_type} startup cancelled.")
            self._reap_dead_locked()
            replacing = spec.port in self._services
            if replacing:
                self._stop_running_locked(spec.port)
            runtime = self._runtime_for(spec)
            log_path = self.log_dir / f"{spec.service_type}-{spec.port}.log"
            try:
                process, log_handle, endpoint = runtime.launch(
                    spec,
                    public_host=self.public_host,
                    log_path=log_path,
                    allow_test_command=self.allow_test_commands,
                )
                self._services[spec.port] = RunningService(spec, process, log_path, log_handle, endpoint)
                runtime.wait_ready(
                    process,
                    endpoint,
                    timeout=float(spec.settings.get("startup_timeout", self.startup_timeout)),
                    cancel_event=cancel_event,
                )
                return endpoint
            except Exception as exc:
                if spec.port in self._services:
                    self._stop_running_locked(spec.port)
                log_tail = self._startup_log_tail(log_path)
                if replacing:
                    raise ServiceStartupError(
                        f"Replacement failed for {spec.service_type} on port {spec.port}; "
                        f"the previous owned process was stopped and was not restored: {exc}{log_tail}"
                    ) from exc
                if isinstance(exc, ServiceStartupError):
                    raise ServiceStartupError(f"{exc}{log_tail}") from exc
                if isinstance(exc, (ValueError, FileNotFoundError)):
                    raise
                if isinstance(exc, ServiceError):
                    raise ServiceStartupError(f"{exc}{log_tail}") from exc
                raise ServiceStartupError(
                    f"Could not start {spec.service_type} on port {spec.port}: {exc}{log_tail}"
                ) from exc

    def stop(self, port: int) -> None:
        with self._lock:
            self._stop_running_locked(port)

    def stop_all(self) -> None:
        with self._lock:
            for port in list(self._services):
                self._stop_running_locked(port)

    def is_running(self, port: int) -> bool:
        with self._lock:
            running = self._services.get(port)
            if running is None:
                return False
            if running.process.poll() is not None:
                self._stop_running_locked(port)
                return False
            return True

    def matches(self, spec: ServiceSpec) -> bool:
        with self._lock:
            running = self._services.get(spec.port)
            return bool(
                running
                and running.process.poll() is None
                and running.spec.service_type == spec.service_type
                and running.spec.settings == spec.settings
            )

    def endpoint_for(self, port: int) -> ServiceEndpoint:
        with self._lock:
            running = self._services.get(port)
            if running is None or running.process.poll() is not None:
                raise ServiceNotRunningError(f"No owned service is running on port {port}.")
            return running.endpoint

    def list_services(self) -> list[ServiceStatus]:
        with self._lock:
            self._reap_dead_locked()
            return [
                ServiceStatus(
                    port=item.spec.port,
                    service_type=item.spec.service_type,
                    running=item.process.poll() is None,
                    settings=dict(item.spec.settings),
                    log_path=str(item.log_path),
                )
                for item in self._services.values()
            ]

    def get_logs(self, port: int, tail: int = 200) -> str:
        with self._lock:
            running = self._services.get(port)
            if running is None:
                raise ServiceNotRunningError(f"No owned service is registered on port {port}.")
            try:
                lines = running.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as exc:
                raise ServiceError(f"Could not read service log: {exc}") from exc
            return "\n".join(lines[-max(1, min(tail, 5000)) :])

    def _stop_running_locked(self, port: int) -> None:
        running = self._services.pop(port, None)
        if running is None:
            return
        try:
            if running.process.poll() is None:
                running.process.terminate()
                try:
                    running.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    running.process.kill()
                    running.process.wait(timeout=5)
        finally:
            running.log_handle.close()

    def _reap_dead_locked(self) -> None:
        for port, running in list(self._services.items()):
            if running.process.poll() is not None:
                self._stop_running_locked(port)
