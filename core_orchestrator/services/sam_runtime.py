"""
sam_runtime.py

In-process SAM3 runtime that manages the singleton Sam3ServerHelper.

Managed SAM does NOT use subprocess.  It loads the model in-process,
keeps it warm, and provides thread-safe inference.

External SAM is handled by storing the base_url and forwarding requests
via HTTP — this module does not manage that path directly; the caller
uses RemoteClientHelper or the LLMInferenceClient pattern.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from .settings_store import SAMServiceSettings

logger = logging.getLogger(__name__)


class SAMError(Exception):
    """Raised when SAM operations fail."""


class SAMRuntime:
    """
    Thread-safe in-process SAM3 model runtime.

    - ``start()`` loads the model via Sam3ServerHelper.
    - ``stop()`` resets the singleton to unload the model.
    - ``restart()`` unloads then reloads.
    - ``predict()`` uses the already-loaded helper.
    - A per-SAM inference lock prevents concurrent model access if the
      underlying implementation is not thread-safe.
    """

    def __init__(self) -> None:
        self._helper: Optional[Any] = None
        self._state: str = "stopped"  # stopped, starting, running, failed, external
        self._inference_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._applied_weights_path: str = ""
        self._last_error: Optional[str] = None
        self._started_at: Optional[str] = None
        self._log_lines: list[str] = []

    # -- Lifecycle ---------------------------------------------------------

    def start(self, settings: SAMServiceSettings) -> dict:
        """Load the SAM3 model in-process."""
        with self._lifecycle_lock:
            if self._state in ("running", "starting"):
                return {"state": self._state, "error": "Already running or starting"}

            if settings.service_mode == "external":
                self._state = "external"
                self._helper = None
                self._last_error = None
                return {"state": "external", "note": "External service - no local model loaded"}

            if not settings.weights_path:
                self._state = "failed"
                self._last_error = "No weights path configured"
                return {"state": "failed", "error": self._last_error}

            self._state = "starting"
            self._last_error = None
            self._log("Starting SAM3 with weights: %s", settings.weights_path)

            try:
                from ..utils.model_host.sam3_server_helper import Sam3ServerHelper

                # Reset singleton before loading to ensure fresh state
                Sam3ServerHelper.reset_singleton()

                helper = Sam3ServerHelper(checkpoint_path=settings.weights_path)
                success = helper.initialize()

                if not success:
                    self._state = "failed"
                    self._last_error = "Failed to initialize SAM3 model"
                    self._log("Failed to initialize SAM3")
                    return {"state": "failed", "error": self._last_error}

                self._helper = helper
                self._applied_weights_path = settings.weights_path
                self._state = "running"
                self._started_at = datetime.now(timezone.utc).isoformat()
                self._log("SAM3 started successfully")
                return {"state": "running", "pid": None}

            except Exception as exc:
                logger.exception("SAM3 start failed")
                self._state = "failed"
                self._last_error = str(exc)
                self._log("Start failed: %s", exc)
                return {"state": "failed", "error": str(exc)}

    def stop(self) -> dict:
        """Unload the model and reset the singleton."""
        with self._lifecycle_lock:
            if self._state == "stopped":
                return {"state": "stopped", "error": "Already stopped"}

            if self._state == "external":
                # Don't change external state — it's config-derived
                return {"state": "external", "note": "External services are managed outside Almost ARCADIA."}

            try:
                if self._helper is not None:
                    from ..utils.model_host.sam3_server_helper import Sam3ServerHelper
                    Sam3ServerHelper.reset_singleton()
                    self._helper = None
            except Exception as exc:
                logger.warning("Error during SAM stop: %s", exc)

            self._state = "stopped"
            self._applied_weights_path = ""
            self._started_at = None
            self._log("SAM3 stopped")
            return {"state": "stopped"}

    def restart(self, settings: SAMServiceSettings) -> dict:
        """Stop then start with new configuration."""
        self.stop()
        return self.start(settings)

    # -- Inference ---------------------------------------------------------

    def predict(
        self,
        frame: Any,
        input_points: Optional[list[list[float]]] = None,
        input_boxes: Optional[list[list[float]]] = None,
    ) -> dict[str, Any]:
        """Run SAM3 prediction using the loaded helper."""
        if self._state != "running" or self._helper is None:
            raise SAMError("SAM3 service is not running")

        with self._inference_lock:
            helper = self._helper
            if input_points:
                result = helper.predict_from_points(frame, input_points, [1] * len(input_points))
            elif input_boxes:
                result = helper.predict_from_box(frame, input_boxes[0])
            else:
                result = helper.predict(frame)

            target_coords = helper.get_target_coordinates(frame, input_points)
            return {
                "masks": result.get("masks", []),
                "scores": result.get("scores", []),
                "bbox": result.get("bbox", []),
                "target_coords": target_coords,
            }

    # -- Status ------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def pid(self) -> Optional[int]:
        return None  # In-process service has no PID

    @property
    def healthy(self) -> bool:
        return self._state == "running" and self._helper is not None

    @property
    def restart_required(self) -> bool:
        return self._state == "running" and bool(self._last_error is None and self._applied_weights_path)

    @property
    def started_at(self) -> Optional[str]:
        return self._started_at

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def applied_weights_path(self) -> str:
        return self._applied_weights_path

    def get_logs(self, tail: int = 50) -> list[str]:
        return self._log_lines[-tail:]

    def _log(self, msg: str, *args: Any) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        line = f"[{ts}] {msg % args if args else msg}"
        self._log_lines.append(line)
        if len(self._log_lines) > 2000:
            self._log_lines = self._log_lines[-2000:]

    def mark_external(self) -> None:
        """Mark this service as externally managed."""
        with self._lifecycle_lock:
            self._state = "external"
            self._helper = None
            self._last_error = None

    def sync_from_config(self, settings: SAMServiceSettings) -> None:
        """
        Synchronize runtime state from saved configuration.

        If service_mode is external, mark the service external.
        If service_mode is managed and service is stopped, leave stopped.
        If the applied config differs from saved, mark restart_required.
        """
        with self._lifecycle_lock:
            if settings.service_mode == "external":
                self._state = "external"
                self._helper = None
            elif self._state == "external" and settings.service_mode == "managed":
                # Was external, now managed — switch to stopped
                self._state = "stopped"

    def check_restart_required(self, settings: SAMServiceSettings) -> bool:
        """Check whether the current saved config differs from applied config."""
        if self._state not in ("running",):
            return False
        return self._applied_weights_path != settings.weights_path