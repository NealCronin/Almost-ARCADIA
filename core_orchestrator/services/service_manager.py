"""
service_manager.py

Manages the lifecycle of model services (LLM, SAM) running on either the
Client or Host machine.

Each service is identified by a unique *service_id* such as ``"client:llm"``,
``"client:sam3"``, ``"host:llm"``, or ``"host:sam3"``.

Concurrency guarantee
---------------------
``_ensure_service()`` atomically creates the runtime AND the lock together.
The lock is never replaced after creation.  All lifecycle methods use the
same pattern::

    runtime, lock = self._ensure_service(service_id)
    with lock:
        ...

Managed LLM services are subprocess-backed via ``ProcessManager``.
Managed SAM services are in-process via ``SAMRuntime``.
External services have no process and no local model.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from .command_builder import build_llm_command
from .llm_client import LLMInferenceClient, LLMResult
from .settings_store import LLMServiceSettings, SAMServiceSettings

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 2000
HEALTH_CHECK_TIMEOUT = 5
DEFAULT_STARTUP_TIMEOUT = 60


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
    applied_config: Optional[dict] = None  # fingerprint of config that was used to start
    sam_runtime: Optional[Any] = None  # SAMRuntime instance for in-process SAM


def _normalize_service_config(settings: LLMServiceSettings | SAMServiceSettings) -> dict:
    """Return a dict of runtime-affecting fields for comparison."""
    if isinstance(settings, LLMServiceSettings):
        return {
            "service_mode": settings.service_mode,
            "executable": settings.executable,
            "model_path": settings.model_path,
            "model_id": settings.model_id,
            "base_url": settings.base_url,
            "api_format": settings.api_format,
            "host": settings.host,
            "port": settings.port,
            "arguments": list(settings.arguments),
        }
    else:
        return {
            "service_mode": settings.service_mode,
            "weights_path": settings.weights_path,
            "base_url": settings.base_url,
            "arguments": list(settings.arguments),
        }


# Runtime-affecting field sets for restart detection
_LLM_RUNTIME_FIELDS = frozenset({
    "service_mode", "executable", "model_path", "model_id",
    "base_url", "api_format", "host", "port", "arguments",
})
_SAM_RUNTIME_FIELDS = frozenset({
    "service_mode", "weights_path", "base_url", "arguments",
})


class ServiceManager:
    """
    Lifecycle manager for model services.

    Thread-safe.  Each service gets its own lock to avoid global contention.
    Locks and runtimes are created together atomically and never replaced.
    """

    def __init__(self) -> None:
        self._services: dict[str, ServiceRuntime] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._llm_client = LLMInferenceClient()

    # -- Atomic creation ----------------------------------------------------

    def _ensure_service(self, service_id: str) -> tuple[ServiceRuntime, threading.Lock]:
        """
        Atomically ensure both the runtime and lock exist for *service_id*.
        Never replaces an existing lock.
        """
        with self._global_lock:
            runtime = self._services.setdefault(service_id, ServiceRuntime())
            lock = self._locks.setdefault(service_id, threading.Lock())
            return runtime, lock

    # -- Public API ---------------------------------------------------------

    def start(
        self,
        service_id: str,
        settings: LLMServiceSettings | SAMServiceSettings,
    ) -> dict:
        """Start a managed service."""
        runtime, lock = self._ensure_service(service_id)
        with lock:
            return self._start_locked(service_id, runtime, settings)

    def stop(self, service_id: str) -> dict:
        """Stop a managed service."""
        runtime, lock = self._ensure_service(service_id)
        with lock:
            return self._stop_locked(service_id, runtime)

    def restart(self, service_id: str, settings: Optional[Any] = None) -> dict:
        """Restart a service. If *settings* is provided, uses the new config."""
        runtime, lock = self._ensure_service(service_id)
        with lock:
            if runtime.state not in (ServiceState.STOPPED, ServiceState.FAILED):
                self._stop_locked(service_id, runtime)
            if settings:
                return self._start_locked(service_id, runtime, settings)
            return {"state": "stopped", "service_id": service_id}

    def status(self, service_id: str) -> dict:
        """Return runtime status for a service."""
        runtime, lock = self._ensure_service(service_id)
        with lock:
            return self._status_dict(service_id, runtime)

    def is_running(self, service_id: str) -> bool:
        runtime, _ = self._ensure_service(service_id)
        return runtime.state in (ServiceState.RUNNING, ServiceState.EXTERNAL)

    def get_logs(self, service_id: str, tail: int = 50) -> list[str]:
        runtime, _ = self._ensure_service(service_id)
        lines = list(runtime.log_lines)
        return lines[-tail:]

    def add_log(self, service_id: str, line: str) -> None:
        runtime, _ = self._ensure_service(service_id)
        runtime.log_lines.append(line)

    def all_status(self) -> dict[str, dict]:
        with self._global_lock:
            ids = list(self._services.keys())
        return {sid: self.status(sid) for sid in ids}

    # -- Config sync and restart detection ---------------------------------

    def sync_configuration(self, service_id: str, settings: LLMServiceSettings | SAMServiceSettings) -> None:
        """
        Synchronize runtime state from saved configuration.

        - External services are marked external.
        - Managed services remain stopped unless explicitly started.
        - If running config differs from saved, mark restart_required.
        """
        runtime, lock = self._ensure_service(service_id)
        with lock:
            if settings.service_mode == "external":
                runtime.state = ServiceState.EXTERNAL
                runtime.healthy = False
                runtime.pid = None
                # For SAM external, mark the runtime
                if runtime.sam_runtime is not None:
                    runtime.sam_runtime.mark_external()
            elif runtime.state == ServiceState.EXTERNAL:
                # Was external, now managed — switch to stopped
                runtime.state = ServiceState.STOPPED

            # Check restart required
            if runtime.state in (ServiceState.RUNNING,):
                current_config = _normalize_service_config(settings)
                if runtime.applied_config and current_config != runtime.applied_config:
                    runtime.restart_required = True
                else:
                    runtime.restart_required = False

    def check_restart_required(
        self, service_id: str, settings: LLMServiceSettings | SAMServiceSettings
    ) -> bool:
        """Check and update restart_required flag."""
        runtime, lock = self._ensure_service(service_id)
        with lock:
            if runtime.state not in (ServiceState.RUNNING,):
                runtime.restart_required = False
                return False
            current = _normalize_service_config(settings)
            if runtime.applied_config and current != runtime.applied_config:
                runtime.restart_required = True
            else:
                runtime.restart_required = False
            return runtime.restart_required

    # -- External services --------------------------------------------------

    def mark_external(self, service_id: str) -> None:
        runtime, lock = self._ensure_service(service_id)
        with lock:
            runtime.state = ServiceState.EXTERNAL
            runtime.healthy = False
            runtime.pid = None

    def mark_external_healthy(self, service_id: str, healthy: bool = True) -> None:
        runtime, lock = self._ensure_service(service_id)
        with lock:
            runtime.state = ServiceState.EXTERNAL
            runtime.healthy = healthy

    # -- LLM inference ------------------------------------------------------

    @property
    def llm_client(self) -> LLMInferenceClient:
        return self._llm_client

    def evaluate_llm(
        self,
        service_id: str,
        settings: LLMServiceSettings,
        prompt: str,
        **kwargs: Any,
    ) -> LLMResult:
        """
        Evaluate a prompt using the configured LLM service.

        For managed services: checks that the service is running, then
        uses the LLM client to call the local HTTP endpoint.
        For external services: uses the LLM client with the configured base_url.
        """
        runtime, lock = self._ensure_service(service_id)
        with lock:
            state = runtime.state

        if state == ServiceState.EXTERNAL:
            return self._llm_client.evaluate(settings, prompt, **kwargs)
        elif state == ServiceState.RUNNING:
            return self._llm_client.evaluate(settings, prompt, **kwargs)
        else:
            from .llm_client import LLMInferenceError
            raise LLMInferenceError(
                f"LLM service {service_id} is not running (state: {state.value})"
            )

    # -- Internal helpers ---------------------------------------------------

    def _start_locked(
        self,
        service_id: str,
        runtime: ServiceRuntime,
        settings: LLMServiceSettings | SAMServiceSettings,
    ) -> dict:
        # Don't start if already running/starting
        if runtime.state in (ServiceState.RUNNING, ServiceState.STARTING):
            return {
                "state": runtime.state.value,
                "service_id": service_id,
                "error": "Service is already running or starting",
            }

        # Determine service type
        is_llm = isinstance(settings, LLMServiceSettings)

        if settings.service_mode != "managed":
            runtime.state = ServiceState.EXTERNAL
            runtime.healthy = False
            return self._status_dict(service_id, runtime)

        # Validate required config
        if is_llm:
            if not settings.executable:
                runtime.state = ServiceState.FAILED
                err = "No executable configured for managed LLM"
                runtime.last_error = err
                return {"state": "failed", "service_id": service_id, "error": err}
        else:
            if not settings.weights_path:
                runtime.state = ServiceState.FAILED
                err = "No weights path configured for managed SAM"
                runtime.last_error = err
                return {"state": "failed", "service_id": service_id, "error": err}

        runtime.state = ServiceState.STARTING
        runtime.last_error = None
        config_snapshot = _normalize_service_config(settings)

        try:
            if is_llm:
                return self._start_llm_subprocess(service_id, runtime, settings, config_snapshot)
            else:
                return self._start_sam_inprocess(service_id, runtime, settings, config_snapshot)
        except Exception as exc:
            logger.exception("Failed to start service %s", service_id)
            runtime.state = ServiceState.FAILED
            runtime.last_error = str(exc)
            runtime.log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] Failed: {exc}")
            return {"state": "failed", "service_id": service_id, "error": str(exc)}

    def _start_llm_subprocess(
        self,
        service_id: str,
        runtime: ServiceRuntime,
        settings: LLMServiceSettings,
        config_snapshot: dict,
    ) -> dict:
        """Start a managed LLM via subprocess."""
        from ..utils.model_host.process_manager import ProcessManager, AlreadyRunningError

        pm = ProcessManager.instance()
        cmd = build_llm_command(settings)

        try:
            pid = pm.start(service_id, cmd)
        except AlreadyRunningError as exc:
            runtime.state = ServiceState.RUNNING
            runtime.pid = pm.get_pid(service_id)
            runtime.log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] Already running: {exc}")
            return self._status_dict(service_id, runtime)

        runtime.pid = pid
        runtime.started_at = datetime.now(timezone.utc).isoformat()

        # Health check
        startup_timeout = settings.startup_timeout_seconds
        healthy = self._wait_for_llm_health(service_id, settings, runtime, timeout=startup_timeout)

        if healthy:
            runtime.state = ServiceState.RUNNING
            runtime.healthy = True
            runtime.restart_required = False
            runtime.applied_config = config_snapshot
            runtime.log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] Started PID {pid}")
        else:
            runtime.state = ServiceState.UNHEALTHY
            runtime.healthy = False
            runtime.last_error = "Health check failed during startup"
            runtime.log_lines.append(
                f"[{datetime.now(timezone.utc).isoformat()}] Started PID {pid} but health check failed"
            )

        return self._status_dict(service_id, runtime)

    def _start_sam_inprocess(
        self,
        service_id: str,
        runtime: ServiceRuntime,
        settings: SAMServiceSettings,
        config_snapshot: dict,
    ) -> dict:
        """Start a managed SAM in-process (no subprocess)."""
        from .sam_runtime import SAMRuntime

        if runtime.sam_runtime is None:
            runtime.sam_runtime = SAMRuntime()

        result = runtime.sam_runtime.start(settings)

        if result.get("state") == "running":
            runtime.state = ServiceState.RUNNING
            runtime.healthy = True
            runtime.pid = None  # In-process, no PID
            runtime.restart_required = False
            runtime.applied_config = config_snapshot
            runtime.started_at = runtime.sam_runtime.started_at
        elif result.get("state") == "external":
            runtime.state = ServiceState.EXTERNAL
        else:
            runtime.state = ServiceState.FAILED if result.get("state") == "failed" else ServiceState.UNHEALTHY
            runtime.healthy = False
            runtime.last_error = result.get("error", "SAM start failed")

        runtime.log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] SAM: {result}")
        return self._status_dict(service_id, runtime)

    def _stop_locked(self, service_id: str, runtime: ServiceRuntime) -> dict:
        if runtime.state == ServiceState.STOPPED:
            return {"state": "stopped", "service_id": service_id, "error": "Already stopped"}

        if runtime.state == ServiceState.EXTERNAL:
            return {"state": "external", "service_id": service_id, "note": "External services are managed outside Almost ARCADIA."}

        runtime.state = ServiceState.STOPPING

        # If SAM in-process, stop it
        if runtime.sam_runtime is not None:
            runtime.sam_runtime.stop()

        # If subprocess, stop it
        if runtime.pid is not None:
            try:
                from ..utils.model_host.process_manager import ProcessManager
                pm = ProcessManager.instance()
                pm.stop(service_id, force=False)
            except Exception as exc:
                logger.exception("Error stopping %s", service_id)

        # Collect final logs from ProcessManager
        try:
            from ..utils.model_host.process_manager import ProcessManager
            pm = ProcessManager.instance()
            stdout, stderr = pm.get_output(service_id)
            for line in stdout[-10:]:
                runtime.log_lines.append(f"[stdout] {line}")
            for line in stderr[-10:]:
                runtime.log_lines.append(f"[stderr] {line}")
        except Exception:
            pass

        runtime.state = ServiceState.STOPPED
        runtime.pid = None
        runtime.healthy = False
        runtime.started_at = None
        runtime.applied_config = None
        runtime.log_lines.append(f"[{datetime.now(timezone.utc).isoformat()}] Stopped")
        return {"state": "stopped", "service_id": service_id}

    def _wait_for_llm_health(
        self,
        service_id: str,
        settings: LLMServiceSettings,
        runtime: ServiceRuntime,
        timeout: int = DEFAULT_STARTUP_TIMEOUT,
    ) -> bool:
        """Poll the LLM health endpoint until ready or timeout."""
        # Check if process exited during startup
        from ..utils.model_host.process_manager import ProcessManager
        pm = ProcessManager.instance()

        if not pm.is_running(service_id):
            runtime.last_error = "Process exited during startup"
            return False

        # Resolve health URL
        base_url = settings.base_url or f"http://{settings.host}:{settings.port}"
        if settings.api_format == "llama-completion":
            health_url = f"{base_url}/health"
        elif settings.api_format in ("openai-chat", "openai-responses"):
            if "/v1" in base_url:
                health_url = f"{base_url}/models"
            else:
                health_url = f"{base_url}/v1/models"
        else:
            health_url = base_url

        timeout = max(timeout, 5)
        deadline = time.time() + timeout

        import requests
        while time.time() < deadline:
            # Check if process died
            if not pm.is_running(service_id):
                stdout, stderr = pm.get_output(service_id)
                runtime.last_error = "Process exited during startup. "
                if stderr:
                    runtime.last_error += "Recent stderr: " + "; ".join(stderr[-5:])
                return False

            try:
                resp = requests.get(health_url, timeout=HEALTH_CHECK_TIMEOUT)
                if 200 <= resp.status_code < 300:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)

        return False

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
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ServiceManager()
    return _manager


def reset_service_manager_for_tests() -> None:
    """Reset the global ServiceManager singleton. Intended for test isolation."""
    global _manager
    with _manager_lock:
        _manager = None