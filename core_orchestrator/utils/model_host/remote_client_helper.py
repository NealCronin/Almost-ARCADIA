"""
remote_client_helper.py

Network utility for client-to-host operations.

Responsibilities
----------------
* Serialize parameters and frames via HTTP POST to remote Host.
* Process JSON responses from the Host's inference endpoints.
* Handle retries and connection errors gracefully.
"""

from __future__ import annotations

import base64
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
    Helper class for communicating with remote Host inference endpoints.

    Attributes
    ----------
    base_url : str
        Base URL of the remote Host (e.g., "http://192.168.1.100:8000").
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.timeout = REQUEST_TIMEOUT

    def _make_request(
        self,
        endpoint: str,
        json_data: Optional[dict] = None,
        files: Optional[dict] = None,
        retries: int = MAX_RETRIES,
    ) -> dict[str, Any]:
        """
        Make a POST request with retry logic.

        Parameters
        ----------
        endpoint : str
            API endpoint path (e.g., "/api/host/evaluate-llm/").
        json_data : dict | None
            JSON payload to send.
        files : dict | None
            Files to upload (for frame data).
        retries : int
            Number of retry attempts.

        Returns
        -------
        dict[str, Any]
            Parsed JSON response.

        Raises
        ------
        RemoteClientError
            If all retries fail.
        """
        url = f"{self.base_url}{endpoint}"

        for attempt in range(retries):
            try:
                resp = self._session.post(
                    url,
                    json=json_data,
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as exc:
                if attempt < retries - 1:
                    delay = RETRY_BACKOFF * (2**attempt)
                    logger.warning(
                        "Request failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        retries,
                        delay,
                        exc,
                    )
                    import time

                    time.sleep(delay)
                else:
                    raise RemoteClientError(
                        f"Failed after {retries} attempts: {exc}"
                    ) from exc

        raise RemoteClientError("Unexpected retry loop exit")

    def evaluate_llm(
        self,
        prompt: str,
        context: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Send an LLM evaluation request to the remote Host.

        Parameters
        ----------
        prompt : str
            The input prompt.
        context : str | None
            Optional context for the evaluation.
        **kwargs
            Additional parameters (max_tokens, temperature, etc.).

        Returns
        -------
        dict[str, Any]
            Response containing the generated text.

        Raises
        ------
        RemoteClientError
            If the request fails.
        """
        payload = {"prompt": prompt}
        if context:
            payload["context"] = context

        # Add generation parameters
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        result = self._make_request("/api/host/evaluate-llm/", json_data=payload)
        return result

    def evaluate_sam3(
        self,
        image: Any,
        input_points: Optional[list[list[float]]] = None,
        input_boxes: Optional[list[list[float]]] = None,
    ) -> dict[str, Any]:
        """
        Send a SAM3 segmentation request to the remote Host.

        Parameters
        ----------
        image : Any
            Input image as numpy array.
        input_points : list[list[float]] | None
            Point prompts.
        input_boxes : list[list[float]] | None
            Box prompts.

        Returns
        -------
        dict[str, Any]
            Response containing segmentation masks and coordinates.

        Raises
        ------
        RemoteClientError
            If the request fails.
        """
        # Encode image as base64 PNG
        cv2 = _get_cv2()
        if cv2 is None:
            raise RemoteClientError("OpenCV not available for image encoding")

        success, png_bytes = cv2.imencode(".png", image)
        if not success:
            raise RemoteClientError("Failed to encode image")

        files = {"image": ("frame.png", png_bytes.tobytes(), "image/png")}
        payload = {}

        if input_points:
            payload["input_points"] = input_points
        if input_boxes:
            payload["input_boxes"] = input_boxes

        result = self._make_request("/api/host/evaluate-sam3/", json_data=payload, files=files)
        return result

    def get_status(self) -> dict[str, Any]:
        """
        Get the remote Host status.

        Returns
        -------
        dict[str, Any]
            Host status information.
        """
        return self._make_request("/api/host/status/")

    def check_connection(self) -> bool:
        """
        Check if the remote Host is reachable.

        Returns
        -------
        bool
            True if the Host responds to status requests.
        """
        try:
            status = self.get_status()
            return "status" in status or "running" in status
        except RemoteClientError:
            return False


# ------------------------------------------------------------------
# OpenCV helpers (lazy import to avoid hard dependency)
# ------------------------------------------------------------------
_cv2 = None


def _get_cv2():
    """Lazy import of OpenCV."""
    global _cv2
    if _cv2 is None:
        try:
            import cv2

            _cv2 = cv2
        except ImportError:
            logger.warning("OpenCV not available")
            return None
    return _cv2


def cv2_imencode(frame: Any) -> tuple[bool, bytes]:
    """
    Encode a numpy array as PNG bytes.

    Parameters
    ----------
    frame : Any
        Input image as numpy array.

    Returns
    -------
    tuple[bool, bytes]
        Success flag and encoded bytes.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        return (False, b"")
    success, buffer = cv2.imencode(".png", frame)
    return (success, buffer.tobytes() if success else b"")

def cv2_imdecode(buf: Any) -> Any:
    """
    Decode PNG bytes to a numpy array.

    Parameters
    ----------
    buf : Any
        Image bytes or array-like.

    Returns
    -------
    Any
        Decoded image as numpy array, or None on failure.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        return None
    try:
        if isinstance(buf, bytes):
            arr = np.frombuffer(buf, dtype=np.uint8)
        else:
            arr = buf
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def image_to_base64(image: Any, format: str = "PNG") -> str:
    """
    Convert a numpy array image to base64 string.

    Parameters
    ----------
    image : Any
        Input image as numpy array.
    format : str
        Image format (PNG, JPEG, etc.).

    Returns
    -------
    str
        Base64-encoded image string.
    """
    cv2 = _get_cv2()
    if cv2 is None:
        raise RemoteClientError("OpenCV not available")

    success, buffer = cv2.imencode(f".{format.lower()}", image)
    if not success:
        raise RemoteClientError("Failed to encode image")

    return base64.b64encode(buffer).decode("utf-8")


def base64_to_image(b64_string: str) -> Optional[Any]:
    """
    Convert base64 string to numpy array image.

    Parameters
    ----------
    b64_string : str
        Base64-encoded image string.

    Returns
    -------
    Any | None
        Decoded image as numpy array, or None on failure.
    """
    try:
        import numpy as np

        data = base64.b64decode(b64_string)
        arr = np.frombuffer(data, dtype=np.uint8)
        cv2 = _get_cv2()
        if cv2 is None:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.debug("Failed to decode base64 image: %s", exc)
        return None


