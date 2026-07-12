"""
inference_service.py

Centralized inference logic shared by both the Django views and the
custom Host HTTP listener.  Transport layers call these functions and
only handle HTTP response formatting.
"""

from __future__ import annotations

import base64
import logging
from typing import Any, Optional

import numpy as np

from .llm_client import LLMInferenceClient, LLMInferenceError, LLMResult
from .settings_store import LLMServiceSettings, SAMServiceSettings, get_settings_store
from .service_manager import get_service_manager

logger = logging.getLogger(__name__)


class ServiceNotRunningError(Exception):
    pass


class InvalidConfigurationError(Exception):
    pass


class InferenceRequestError(Exception):
    pass


class ExternalServiceError(Exception):
    pass


def _client():
    """Return a shared LLMInferenceClient."""
    return get_service_manager().llm_client


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------
def evaluate_host_llm(
    request_data: dict[str, Any],
    store=None,
    sm=None,
) -> dict:
    """
    Evaluate an LLM prompt using Host-owned configuration.

    Returns a stable dict::

        {"content": "...", "model_id": "...", "usage": null, "metadata": {}}

    Raises typed exceptions that transport layers convert to structured responses.
    """
    store = store or get_settings_store()
    sm = sm or get_service_manager()

    prompt = request_data.get("prompt", "")
    if not prompt:
        raise InferenceRequestError("prompt is required")

    max_tokens = request_data.get("max_tokens", 512)
    temperature = request_data.get("temperature", 0.7)

    # Validate generation params
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise InferenceRequestError("max_tokens must be a positive integer")
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > 2:
        raise InferenceRequestError("temperature must be between 0 and 2")

    settings = store.load()
    llm_cfg = settings.host.llm

    # Sync external state
    sm.sync_configuration("host:llm", llm_cfg)

    llm_status = sm.status("host:llm")

    if llm_status["state"] in ("running", "external"):
        result = sm.evaluate_llm(
            "host:llm",
            llm_cfg,
            prompt,
            context=request_data.get("context"),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        log_request_safe("/api/host/evaluate-llm/", {"prompt_length": len(prompt)})
        return {
            "content": result.content,
            "model_id": result.model_id,
            "usage": result.usage,
            "metadata": result.metadata,
        }
    else:
        raise ServiceNotRunningError("Host LLM service is not running")


# ---------------------------------------------------------------------------
# SAM inference
# ---------------------------------------------------------------------------
def evaluate_host_sam3(
    request_data: dict[str, Any],
    store=None,
    sm=None,
) -> dict:
    """
    Evaluate a SAM3 segmentation request using Host-owned configuration.

    For managed SAM: requires state running; uses the in-process SAMRuntime.
    For external SAM: forwards to the configured base_url.
    """
    store = store or get_settings_store()
    sm = sm or get_service_manager()

    frame_b64 = request_data.get("frame_b64")
    if not frame_b64:
        raise InferenceRequestError("frame_b64 is required")

    # Decode and validate image
    frame = _decode_image(frame_b64)
    if frame is None:
        raise InferenceRequestError("Invalid image data")

    input_points = request_data.get("input_points")
    input_boxes = request_data.get("input_boxes")

    settings = store.load()
    sam_cfg = settings.host.sam3

    # Sync external state
    sm.sync_configuration("host:sam3", sam_cfg)

    sam_status = sm.status("host:sam3")

    if sam_status["state"] == "running":
        # Use the in-process SAM runtime
        runtime, lock = sm._ensure_service("host:sam3")
        with lock:
            sam_rt = runtime.sam_runtime
            if sam_rt is None:
                raise ServiceNotRunningError("SAM runtime not initialized")

            result = sam_rt.predict(frame, input_points=input_points, input_boxes=input_boxes)
            log_request_safe("/api/host/evaluate-sam3/", {"frame_shape": str(frame.shape)})
            return result

    elif sam_status["state"] == "external":
        # External SAM — forward to configured base_url
        import requests
        try:
            # Encode the frame
            import cv2
            success, png_bytes = cv2.imencode(".png", frame)
            if not success:
                raise ExternalServiceError("Failed to encode image for external SAM")

            payload = {"frame_b64": base64.b64encode(png_bytes).decode("utf-8")}
            if input_points:
                payload["input_points"] = input_points
            if input_boxes:
                payload["input_boxes"] = input_boxes

            resp = requests.post(
                f"{sam_cfg.base_url}/predict",
                json=payload,
                timeout=sam_cfg.request_timeout_seconds,
            )
            resp.raise_for_status()
            result = resp.json()
            log_request_safe("/api/host/evaluate-sam3/", {"frame_shape": str(frame.shape)})
            return result

        except requests.RequestException as exc:
            raise ExternalServiceError(f"External SAM request failed: {exc}") from exc

    else:
        raise ServiceNotRunningError("Host SAM service is not running")


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------
def _decode_image(frame_b64: str) -> Optional[np.ndarray]:
    """Decode a base64-encoded image with strict validation."""
    try:
        raw = base64.b64decode(frame_b64, validate=True)
    except Exception:
        return None

    try:
        import cv2
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return None

        # Validate dimensions
        h, w = frame.shape[:2]
        if h > 10000 or w > 10000 or h * w > 50_000_000:
            return None
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return None

        return frame
    except Exception:
        return None


def log_request_safe(endpoint: str, details: dict) -> None:
    """Safe request logging that doesn't crash if called outside Django context."""
    try:
        from ..views.pages import log_request
        log_request(endpoint, details)
    except Exception:
        pass