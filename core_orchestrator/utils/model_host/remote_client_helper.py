"""
remote_client_helper.py

Client-side helper for making requests to a remote Host running SAM3/LLM inference.
This module provides the RemoteClientHelper class for communicating with
the Host API server over HTTP.
"""

from __future__ import annotations

import base64
import logging
import json
from typing import Any, Optional

import numpy as np
import requests

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
REQUEST_TIMEOUT = 60  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF = 1.0  # seconds


# ------------------------------------------------------------------
# Exception
# ------------------------------------------------------------------
class RemoteClientError(Exception):
    """Raised when a remote request fails."""


# ------------------------------------------------------------------
# Main helper
# ------------------------------------------------------------------
class RemoteClientHelper:
    """
    Client helper for making requests to a remote Host.

    This class handles HTTP communication with the Host API server,
    including image encoding, request retries, and error handling.

    Attributes
    ----------
    base_url : str
        Base URL of the Host API server.
    timeout : int
        Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    # -- Public methods -----------------------------------------------------

    def evaluate_llm(
        self,
        prompt: str,
        context: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Send an LLM evaluation request to the remote Host (POST)."""
        payload = {"prompt": prompt}
        if context:
            payload["context"] = context
        payload.update({k: v for k, v in kwargs.items() if v is not None})
        return self._make_request("POST", "/api/host/evaluate-llm/", json_data=payload)

    def evaluate_sam3(
        self,
        image: Any,
        input_points: Optional[list[list[float]]] = None,
        input_boxes: Optional[list[list[float]]] = None,
    ) -> dict[str, Any]:
        """
        Send a SAM3 segmentation request to the remote Host (POST).

        NOTE: The Host manages its own SAM3 weights path configuration.
        The client does NOT send weights_path - it is read from the Host's
        configuration on the server side.
        """
        cv2 = _get_cv2()
        if cv2 is None:
            raise RemoteClientError("OpenCV not available for image encoding")

        success, png_bytes = cv2.imencode(".png", image)
        if not success:
            raise RemoteClientError("Failed to encode image")

        frame_b64 = base64.b64encode(png_bytes).decode("utf-8")

        payload = {"frame_b64": frame_b64}
        if input_points:
            payload["input_points"] = input_points
        if input_boxes:
            payload["input_boxes"] = input_boxes

        return self._make_request("POST", "/api/host/evaluate-sam3/", json_data=payload)

    def get_status(self) -> dict[str, Any]:
        """Get the remote Host status (GET)."""
        return self._make_request("GET", "/api/host/status/")

    def check_connection(self) -> bool:
        """Check if the remote Host is reachable."""
        try:
            status = self.get_status()
            return "status" in status or "running" in status
        except RemoteClientError:
            return False

    # -- Internal helpers ---------------------------------------------------

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[dict[str, Any]] = None,
        files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Make an HTTP request to the Host API with retries.

        Parameters
        ----------
        method : str
            HTTP method ("GET" or "POST").
        endpoint : str
            API endpoint path.
        json_data : dict | None
            JSON payload for the request.
        files : dict | None
            Files to upload.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response from the server.

        Raises
        ------
        RemoteClientError
            If the request fails after retries.
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                if method.upper() == "GET":
                    resp = self._session.get(url, timeout=self.timeout)
                else:
                    resp = self._session.post(
                        url,
                        json=json_data,
                        files=files,
                        timeout=self.timeout,
                    )
                resp.raise_for_status()
                try:
                    return resp.json()
                except (ValueError, json.JSONDecodeError) as exc:
                    raise RemoteClientError(f"Invalid JSON response: {exc}") from exc

            except requests.RequestException as exc:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BACKOFF * (2**attempt)
                    logger.warning(
                        "Request failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        MAX_RETRIES,
                        delay,
                        exc,
                    )
                    import time
                    time.sleep(delay)
                else:
                    raise RemoteClientError(
                        f"Failed after {MAX_RETRIES} attempts: {exc}"
                    ) from exc

        raise RemoteClientError("Unexpected retry loop exit")


# ------------------------------------------------------------------
# OpenCV helpers (lazy import to avoid hard dependency)
# ------------------------------------------------------------------
_cv2 = None


def _get_cv2():
    """Lazy import of OpenCV."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2 as _cv2_module
            _cv2 = _cv2_module
        except ImportError:
            logger.warning("OpenCV not available")
            _cv2 = None
    return _cv2


def cv2_imencode(frame: Any) -> tuple[bool, bytes]:
    """Encode an image frame to PNG bytes."""
    cv2 = _get_cv2()
    if cv2 is None:
        return (False, b"")
    try:
        success, buffer = cv2.imencode(".png", frame)
        return (success, buffer.tobytes() if success else b"")
    except Exception:
        return (False, b"")


def cv2_imdecode(buf: Any) -> Any:
    """
    Decode PNG bytes to an image frame.

    Uses module-level numpy import for safety.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        return None
    try:
        arr = np.frombuffer(buf, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def image_to_base64(
    image: Any,
    format: str = "png",
    quality: int = 95,
) -> str:
    """Encode an image to a base64 string."""
    cv2 = _get_cv2()
    if cv2 is None:
        raise RemoteClientError("OpenCV not available")
    try:
        if format.lower() == "png":
            success, buffer = cv2.imencode(".png", image)
        else:
            success, buffer = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
        if not success:
            raise RemoteClientError("Failed to encode image")
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as exc:
        raise RemoteClientError(f"Image encoding failed: {exc}") from exc


def base64_to_image(b64_string: str) -> Optional[Any]:
    """Decode a base64 string to an image."""
    try:
        decoded = base64.b64decode(b64_string)
        arr = np.frombuffer(decoded, dtype=np.uint8)
        cv2 = _get_cv2()
        if cv2 is None:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None