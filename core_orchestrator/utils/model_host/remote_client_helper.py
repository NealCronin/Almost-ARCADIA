"""
remote_client_helper.py

Remote inference client that talks to a backend serving LLM and SAM3
models over HTTP.

Responsibilities
----------------
* Send prompts to a remote LLM endpoint.
* Send frames (with optional points/boxes) to a remote SAM3 endpoint.
* Query host status of the remote service.
* Handle base64 encoding/decoding of images via OpenCV.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
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
    HTTP client for remote LLM and SAM3 inference endpoints.

    Parameters
    ----------
    base_url : str
        Base URL of the remote service (e.g. ``http://remote-host:8000``).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    def evaluate_llm(self, prompt: str, **kwargs: Any) -> str:
        """
        Send *prompt* to the remote LLM endpoint.

        Parameters
        ----------
        prompt : str
        **kwargs
            Extra keys forwarded to the endpoint (temperature, etc.).

        Returns
        -------
        str
            Generated text.
        """
        payload = {"prompt": prompt, **kwargs}
        resp = self._post("/llm/evaluate", payload)
        return resp.get("text", resp.get("generation", ""))

    # ------------------------------------------------------------------
    # SAM3
    # ------------------------------------------------------------------
    def evaluate_sam3(
        self,
        frame: Any,
        points: Optional[list] = None,
        boxes: Optional[list] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Send a frame with optional prompts to the remote SAM3 endpoint.

        Parameters
        ----------
        frame : numpy.ndarray or bytes-like
            Input image.
        points : list of [x, y], optional
        boxes : list of [x1, y1, x2, y2], optional
        **kwargs

        Returns
        -------
        dict
            ``{"boxes": [...], "masks": [[...]], "scores": [...]}``
        """
        # Encode frame to base64
        frame_b64 = self._encode_frame(frame)

        payload: dict[str, Any] = {
            "frame": frame_b64,
            "points": points,
            "boxes": boxes,
            **kwargs,
        }
        resp = self._post("/sam3/evaluate", payload)
        return {
            "boxes": resp.get("boxes", []),
            "masks": resp.get("masks", []),
            "scores": resp.get("scores", []),
        }

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def host_status(self) -> dict[str, Any]:
        """
        Query the health / status endpoint of the remote service.

        Returns
        -------
        dict
            Service status information.
        """
        resp = self._get("/status")
        return resp

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any], retries: int = MAX_RETRIES) -> dict[str, Any]:
        """POST with retries."""
        url = f"{self.base_url}{path}"
        for attempt in range(1, retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("POST %s attempt %d/%d failed: %s", path, attempt, retries, exc)
                if attempt < retries:
                    import time

                    time.sleep(RETRY_BACKOFF * attempt)
        raise RemoteClientError(f"POST {path} failed after {retries} retries")

    def _get(self, path: str, retries: int = MAX_RETRIES) -> dict[str, Any]:
        """GET with retries."""
        url = f"{self.base_url}{path}"
        for attempt in range(1, retries + 1):
            try:
                resp = self._session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning("GET %s attempt %d/%d failed: %s", path, attempt, retries, exc)
                if attempt < retries:
                    import time

                    time.sleep(RETRY_BACKOFF * attempt)
        raise RemoteClientError(f"GET {path} failed after {retries} retries")

    @staticmethod
    def _encode_frame(frame: Any) -> str:
        """
        Encode a numpy image array to base64 via OpenCV.

        Parameters
        ----------
        frame : numpy.ndarray
            Image array (H, W, C).

        Returns
        -------
        str
            Base64-encoded PNG string.
        """
        import numpy as np

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        success, buf = cv2_imencode(frame)
        if not success:
            raise RemoteClientError("Failed to encode frame with OpenCV")
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def decode_frame(b64: str) -> Any:
        """
        Decode a base64 string back to a numpy image array.

        Parameters
        ----------
        b64 : str

        Returns
        -------
        numpy.ndarray
        """
        raw = base64.b64decode(b64)
        import numpy as np

        arr = np.frombuffer(raw, dtype=np.uint8)
        return cv2_imdecode(arr)


# ------------------------------------------------------------------
# OpenCV helpers (lazy import to avoid hard dependency)
# ------------------------------------------------------------------
_cv2 = None


def _get_cv2() -> Any:
    global _cv2
    if _cv2 is None:
        try:
            import cv2 as _cv2_module

            _cv2 = _cv2_module
        except ImportError:
            raise RemoteClientError("opencv-python is required for frame encoding/decoding")
    return _cv2


def cv2_imencode(frame: Any) -> tuple[bool, bytes]:
    """Encode a numpy array as PNG bytes."""
    cv2 = _get_cv2()
    return cv2.imencode(".png", frame)


def cv2_imdecode(buf: Any) -> Any:
    """Decode PNG bytes back to a numpy array."""
    cv2 = _get_cv2()
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
