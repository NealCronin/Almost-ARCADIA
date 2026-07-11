"""
service_manager.py

Manages the lifecycle of model services (LLM, SAM) running on either the
Client or Host machine.

Each service is identified by a unique *service_id* such as ``"client:llm"``,
``"client:sam3"``, ``"host:llm"``, or ``"host:sam3"``.

Managed services are started/stopped via ``ProcessManager``.
External services are not touched — the manager only tracks their
configured state and provides health checks.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .command_builder import build_llm_command, build_sam_command
from .settings_store import LLMServiceSettings, SAMServiceSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_LOG_LINES = 2000
HEALTH_CHECK_TIMEOUT = 5  # seconds
DEFAULT_STARTUP_TIMEOUT = 60  # seconds


# ---------------------------------------------------------------------------
# Service state
# ---------------------------------------------------------------------------
class ServiceState(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    FAILED = "failed"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


@dataclass
class ServiceRuntime:
    """Runtime state for a single service instance."""
    state: ServiceState = ServiceState.STOPPED
    pid: Optional[int] = None
    healthy: bool = False
    restart_required: bool = False
    started_at: Optional[str] = None
    last_error: Optional[str] = None
    log_lines: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LOG_LINES))


# ---------------------------------------------------------------------------
# Service manager
# ---------------------------------------------------------------------------
class ServiceManager:
    """
    Lifecycle manager for model services.

    Thread-safe.  Each service gets its own lock to avoid global contention.
    """

    def __init__(self) -> None:
        self._services: dict[str, ServiceRuntime] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._pm: Any = None  # lazy import of ProcessManager

    # -- Public API ---------------------------------------------------------

    def start(
        self,
        service_id: str,
        settings: LLMServiceSettings | SAMServiceSettings,
    ) -> dict:
        """
        Start a managed service.

        Returns a status dict with ``state``, ``pid``, and optional ``error``.
        """
        lock = self._get_lock(service_id)
        with lock:
            return self._start_locked(service_id, settings)

    def stop(self, service_id: str) -> dict:
        """Stop a managed service."""
        lock = self._get_lock(service_id)
        with lock:
            return self._stop_locked(service_id)

    def restart(self, service_id: str, settings: Optional[Any] = None) -> dict:
        """
        Restart a service.

        If ``settings`` is provided it overrides the current configuration.
        """
        lock = self._get_lock(service_id)
        with lock:
            runtime = self._get_runtime(service_id)
            # Stop
            if runtime.state not in (ServiceState.STOPPED, ServiceState.FAILED):
                self._stop_locked(service_id)
            if settings:
                return self._start_locked(service_id, settings)
            return {"state": "stopped", "service_id": service_id}

    def status(self, service_id: str) -> dict:
        """Return runtime status for a service."""
        runtime = self._get_runtime(service_id)
        return self._status_dict(service_id, runtime)

    def is_running(self, service_id: str) -> bool:
        """Check if the service is considered running."""
        runtime = self._get_runtime(service_id)
        return runtime.state in (ServiceState.RUNNING, ServiceState.EXTERNAL)

    def get_logs(self, service_id: str, tail: int = 50) -> list[str]:
        """Return the last *tail* log lines for a service."""
        runtime = self._get_runtime(service_id)
        lines = list(runtime.log_lines)
        return lines[-tail:]

    def add_log(self, service_id: str, line: str) -> None:
        """Append a line to the service log."""
        runtime = self._get_runtime(service_id)
        runtime.log_lines.append(line)

    def mark_restart_required(self, service_id: str) -> None:
        """Mark a service as needing a restart to apply new config."""
        runtime = self._get_runtime(service_id)
        runtime.restart_required = True

    def clear_restart_required(self, service_id: str) -> None:
        """Clear the restart-required flag."""
        runtime = self._get_runtime(service_id)
        runtime.restart_required = False

    def all_status(self) -> dict[str, dict]:
        """Return status for all tracked services."""
        with self._global_lock:
            ids = list(self._services.keys())
        return {sid: self.status(sid) for sid in ids}

    # -- External services --------------------------------------------------

    def mark_external(self, service_id: str) -> None:
        """Mark a service as externally managed (no process to control)."""
        runtime = self._get_runtime(service_id)
        runtime.state = ServiceState.EXTERNAL
        runtime.healthy = False
        runtime.pid = None

    def mark_external_healthy(self, service_id: str, healthy: bool = True) -> None:
        """Update health status of an external service."""
        runtime = self._get_runtime(service_id)
        runtime.state = ServiceState.EXTERNAL
        runtime.healthy = healthy

    # -- Internal helpers ---------------------------------------------------

    def _start_locked(
        self,
        service_id: str,
        settings: LLMServiceSettings | SAMServiceSettings,
    ) -> dict:
        runtime = self._get_runtime(service_id)

        # Don't start if already running/starting
        if runtime.state in (ServiceState.RUNNING, ServiceState.STARTING):
            return {
                "state": runtime.state.value,
                "service_id": service_id,
                "error": "Service is already running or starting",
            }

        if isinstance(settings, LLMServiceSettings):
            if settings.service_mode != "managed":
                runtime.state = ServiceState.EXTERNAL
                runtime.healthy = False
                return self._status_dict(service_id, runtime)
            if not settings.executable and not settings.model_path:
                runtime.state = ServiceState.FAILED
                err = "No executable or model path configured"
                runtime.last_error = err
                return {"state": "failed", "service_id": service_id, "error": err}
        else:
            if settings.service_mode != "managed":
                runtime.state = ServiceState.EXTERNAL
                runtime.healthy = False
                return self._status_dict(service_id, runtime)
            if not settings.weights_path:
                runtime.state = ServiceState.FAILED
                err = "No weights path configured"
                runtime.last_error = err
                return {"state": "failed", "service_id": service_id, "error": err}

        runtime.state = ServiceState.STARTING
        runtime.last_error = None

        try:
            from ..utils.model_host.process_manager import ProcessManager  # noqa: PLC0415

            pm = ProcessManager.instance()

            if isinstance(settings, LLMServiceSettings):
                cmd = build_llm_command(settings)
            else:
                cmd = build_sam_command(settings)

            pid = pm.start(service_id, cmd)
            runtime.pid = pid
            runtime.started_at = datetime.now(timezone.utc).isoformat()

            # Wait for health check
            startup_timeout = getattr(settings, "startup_timeout_seconds", DEFAULT_STARTUP_TIMEOUT)
            healthy = self._wait_for_health(service_id, settings, timeout=startup_timeout)

            if healthy:
                runtime.state = ServiceState.RUNNING
                runtime.healthy = True
                runtime.restart_required = False
                runtime.log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] Started PID {pid}")
            else:
                runtime.state = ServiceState.UNHEALTHY
                runtime.healthy = False
                runtime.last_error = "Service started but health check failed"
                # Don't stop the process - user can inspect logs
                runtime.log_lines.append(
                    f"[{datetime.now(timezone.utc).isoformat()}] Started PID {pid} but health check failed"
                )

            return self._status_dict(service_id, runtime)

        except Exception as exc:
            logger.exception("Failed to start service %s", service_id)
            runtime.state = ServiceState.FAILED
            runtime.last_error = str(exc)
            runtime.log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] Failed: {exc}")
            return {"state": "failed", "service_id": service_id, "error": str(exc)}

    def _stop_locked(self, service_id: str) -> dict:
        runtime = self._get_runtime(service_id)

        if runtime.state == ServiceState.STOPPED:
            return {"state": "stopped", "service_id": service_id, "error": "Already stopped"}

        if runtime.state == ServiceState.EXTERNAL:
            runtime.state = ServiceState.EXTERNAL
            return {"state": "external", "service_id": service_id, "note": "External service cannot be stopped"}

        runtime.state = ServiceState.STOPPING
        try:
            from ..utils.model_host.process_manager import ProcessManager  # noqa: PLC0415

            pm = ProcessManager.instance()
            pm.stop(service_id, force=False)
        except Exception as exc:
            logger.exception("Error stopping %s", service_id)

        runtime.state = ServiceState.STOPPED
        runtime.pid = None
        runtime.healthy = False
        runtime.started_at = None
        runtime.log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] Stopped")
        return {"state": "stopped", "service_id": service_id}

    def _wait_for_health(
        self,
        service_id: str,
        settings: LLMServiceSettings | SAMServiceSettings,
        timeout: int = DEFAULT_STARTUP_TIMEOUT,
    ) -> bool:
        """Poll the service health endpoint until it responds or timeout."""
        import requests  # noqa: PLC0415

        base_url = settings.base_url or f"http://{settings.host}:{settings.port}" if hasattr(settings, "host") else ""
        if not base_url:
            # No health check possible; assume it's running after a brief delay
            time.sleep(2)
            return True

        timeout = max(timeout, 5)
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = requests.get(
                    f"{base_url}/health" if "llama-completion" in getattr(settings, "api_format", "") else base_url,
                    timeout=HEALTH_CHECK_TIMEOUT,
                )
                if resp.status_code < 500:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)

        return False

    def _get_runtime(self, service_id: str) -> ServiceRuntime:
        """Get or create the runtime record for a service."""
        with self._global_lock:
            if service_id not in self._services:
                self._services[service_id] = ServiceRuntime()
                self._locks[service_id] = threading.Lock()
            return self._services[service_id]

    def _get_lock(self, service_id: str) -> threading.Lock:
        with self._global_lock:
            if service_id not in self._locks:
                self._locks[service_id] = threading.Lock()
            return self._locks[service_id]

    @staticmethod
    def _status_dict(service_id: str, runtime: ServiceRuntime) -> dict:
        return {
            "service_id": service_id,
            "state": runtime.state.value,
            "pid": runtime.pid,
            "healthy": runtime.healthy,
            "restart_required": runtime.restart_required,
            "started_at": runtime.started_at,
            "last_error": runtime.last_error,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_manager: Optional[ServiceManager] = None
_manager_lock = threading.Lock()


def get_service_manager() -> ServiceManager:
    """Return the application-global ServiceManager (created on first call)."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ServiceManager()
    return _manager