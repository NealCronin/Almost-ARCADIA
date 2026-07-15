"""SAM 3 segmentation runtime.

Launches a SAM 3 segmentation model as a standalone HTTP service.
Mirrors the LLMRuntime pattern: validate → build process spec → launch → poll readiness.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable

from arcadia.process import ProcessLauncher, ProcessSpec, RunningProcess
from arcadia.contracts import RunningService, ServiceEndpoint, ServiceSpec, ModelSpec

logger = logging.getLogger(__name__)

_SUPPORTED_SETTINGS = frozenset({
    "host",
    "device",
    "half_precision",
    "default_confidence",
    "extra_args",
})


class SAMRuntimeError(RuntimeError):
    """Raised when the SAM runtime fails to start or stop."""


class SAMRuntime:
    """Launches a SAM 3 segmentation model as a standalone HTTP service.

    Mirrors the LLMRuntime pattern: validate → build process spec → launch → poll readiness.
    """

    def __init__(
        self,
        process_launcher: ProcessLauncher,
        python_executable: Path | None = None,
        readiness_probe: Callable[[str, int], bool] | None = None,
        startup_timeout: float = 120.0,
        poll_interval: float = 0.25,
    ) -> None:
        self._process_launcher = process_launcher
        self._python_executable = python_executable or Path(__import__("sys").executable)
        self._readiness_probe = readiness_probe or self._default_readiness_probe
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, spec: ServiceSpec) -> RunningService:
        """Start the SAM 3 segmentation service.

        Raises SAMRuntimeError on any failure (validation, launch, or readiness).
        """
        self._validate(spec)
        checkpoint_path = self._resolve_checkpoint_path(spec.model)
        proc_spec = self._build_process_spec(spec, checkpoint_path)
        running_process = self._process_launcher.start(proc_spec)
        host = spec.settings.get("host", "127.0.0.1")
        try:
            self._wait_for_service(running_process, host, spec.port)
        except Exception:
            self._process_launcher.stop(running_process)
            raise
        return RunningService(
            spec=spec,
            endpoint=ServiceEndpoint(
                host=host,
                port=spec.port,
                service_type="segmentation",
            ),
            runtime_handle=running_process,
        )
    def stop(self, service: RunningService) -> None:
        """Stop a previously started SAM 3 segmentation service."""
        if not hasattr(service.runtime_handle, "stop"):
            raise SAMRuntimeError("Invalid runtime handle")
        self._process_launcher.stop(service.runtime_handle)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self, spec: ServiceSpec) -> None:
        if spec.service_type != "segmentation":
            raise SAMRuntimeError(
                f"Service type must be 'segmentation', got '{spec.service_type}'"
            )
        if spec.model is None:
            raise SAMRuntimeError("Model is required for segmentation runtime")
        if spec.model.local_path is None:
            raise SAMRuntimeError("Model.local_path is required for SAM runtime")
        if not (1 <= spec.port <= 65535):
            raise SAMRuntimeError(f"Port must be 1-65535, got {spec.port}")

        for key, value in spec.settings.items():
            if key not in _SUPPORTED_SETTINGS:
                raise SAMRuntimeError(f"Unknown setting: {key}")

            if key == "host":
                if not isinstance(value, str) or not value:
                    raise SAMRuntimeError("host must be a non-empty string")
            elif key == "device":
                if not isinstance(value, str) or not value:
                    raise SAMRuntimeError("device must be a non-empty string")
            elif key == "half_precision":
                if not isinstance(value, bool):
                    raise SAMRuntimeError("half_precision must be a bool")
            elif key == "default_confidence":
                if not isinstance(value, (int, float)):
                    raise SAMRuntimeError("default_confidence must be a number")
                if not (0.0 <= value <= 1.0):
                    raise SAMRuntimeError("default_confidence must be between 0.0 and 1.0")
            elif key == "extra_args":
                if not isinstance(value, list):
                    raise SAMRuntimeError("extra_args must be a list")
                for item in value:
                    if not isinstance(item, str):
                        raise SAMRuntimeError("extra_args items must be strings")

    def _resolve_checkpoint_path(self, model: ModelSpec) -> Path:
        if model.local_path is None:
            raise SAMRuntimeError("ModelSpec.local_path is required for SAM runtime")
        path = Path(model.local_path).expanduser()
        if not path.exists() or not path.is_file():
            raise SAMRuntimeError(f"Checkpoint not found: {path}")
        return path

    # ------------------------------------------------------------------
    # Process spec
    # ------------------------------------------------------------------

    def _build_process_spec(self, spec: ServiceSpec, checkpoint_path: Path) -> ProcessSpec:
        command = [
            str(self._python_executable),
            "-m", "arcadia.runtimes.sam_server",
            "--checkpoint", str(checkpoint_path),
            "--host", spec.settings.get("host", "127.0.0.1"),
            "--port", str(spec.port),
        ]
        if "device" in spec.settings:
            command.extend(["--device", spec.settings["device"]])
        if "half_precision" in spec.settings:
            command.extend(["--half-precision", "true" if spec.settings["half_precision"] else "false"])
        if "default_confidence" in spec.settings:
            command.extend(["--default-confidence", str(spec.settings["default_confidence"])])
        if "extra_args" in spec.settings:
            command.extend(spec.settings["extra_args"])
        return ProcessSpec(command=command)

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    @staticmethod
    def _default_readiness_probe(host: str, port: int) -> bool:
        import urllib.request
        import json

        url = f"http://{host}:{port}/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("status") == "ready"
        except Exception:
            return False

    def _wait_for_service(self, running_process: RunningProcess, host: str, port: int) -> None:
        start_time = time.monotonic()
        while time.monotonic() - start_time < self._startup_timeout:
            if not self._process_launcher.is_running(running_process):
                stderr = self._process_launcher.recent_stderr(running_process)
                raise SAMRuntimeError(
                    f"Process exited unexpectedly. Stderr: {stderr}"
                )
            if self._readiness_probe(host, port):
                return
            time.sleep(self._poll_interval)
        raise SAMRuntimeError(
            f"Service did not become ready within {self._startup_timeout}s"
        )