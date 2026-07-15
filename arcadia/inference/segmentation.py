"""Segmentation client for SAM 3 services.

Sends segmentation requests to a running SAM 3 HTTP service and returns
SegmentationResult objects with decoded NumPy boolean masks.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any, Callable

import numpy as np
from PIL import Image

from arcadia.contracts import ServiceEndpoint, SegmentationRequest, SegmentationResult

logger = logging.getLogger(__name__)


class SegmentationClientError(RuntimeError):
    """Raised when the segmentation client fails."""


class SegmentationClient:
    """Client for SAM 3 segmentation services.

    Sends segmentation requests to a running SAM 3 HTTP service and returns
    SegmentationResult objects with decoded NumPy boolean masks.
    """

    def __init__(
        self,
        timeout: float = 120.0,
        urlopen: Callable[..., Any] | None = None,
    ) -> None:
        self._timeout = timeout
        import urllib.request
        self._urlopen = urlopen or urllib.request.urlopen

    def segment(
        self,
        endpoint: ServiceEndpoint,
        request: SegmentationRequest,
        confidence: float | None = None,
    ) -> SegmentationResult:
        """Send a segmentation request and return the result.

        Args:
            endpoint: The service endpoint to connect to.
            request: The segmentation request (image bytes and prompt).
            confidence: Optional confidence threshold override.

        Returns:
            SegmentationResult with decoded masks.

        Raises:
            SegmentationClientError: On any failure.
        """
        # Validate endpoint
        if endpoint.service_type != "segmentation":
            raise SegmentationClientError(
                f"Endpoint service type must be 'segmentation', got '{endpoint.service_type}'"
            )

        # Validate request
        if not request.image:
            raise SegmentationClientError("Empty image bytes")

        # Normalize prompt to list[str]
        if isinstance(request.prompt, str):
            prompts = [request.prompt]
        else:
            prompts = list(request.prompt)

        if not prompts:
            raise SegmentationClientError("Empty prompt")

        # Base64 encode image
        image_b64 = base64.b64encode(request.image).decode("ascii")

        # Build JSON body
        body = {
            "image_base64": image_b64,
            "prompts": prompts,
        }
        if confidence is not None:
            body["confidence"] = confidence

        # Send request
        url = f"http://{endpoint.host}:{endpoint.port}/v1/segment"
        try:
            response = self._send_request(url, body)
        except SegmentationClientError:
            raise
        except Exception as e:
            raise SegmentationClientError(f"Request failed: {e}") from e

        # Parse response
        if not isinstance(response, dict):
            raise SegmentationClientError("Invalid response: not a dict")

        predictions = response.get("predictions", [])
        source_width = response.get("source_width", 0)
        source_height = response.get("source_height", 0)

        # Decode masks
        masks = []
        labels = []
        confidences = []
        bounding_boxes = []

        for pred in predictions:
            label = pred.get("label", "")
            confidence_val = pred.get("confidence", 0.0)
            bounding_box = pred.get("bounding_box", [])
            mask_dict = pred.get("mask", {})

            try:
                mask = self._decode_mask(mask_dict)
            except SegmentationClientError:
                raise
            except Exception as e:
                raise SegmentationClientError(f"Failed to decode mask: {e}") from e

            masks.append(mask)
            labels.append(label)
            confidences.append(confidence_val)
            bounding_boxes.append(bounding_box)

        # Validate alignment
        if not (len(masks) == len(labels) == len(confidences) == len(bounding_boxes)):
            raise SegmentationClientError(
                f"Misaligned data: {len(masks)} masks, {len(labels)} labels, "
                f"{len(confidences)} confidences, {len(bounding_boxes)} boxes"
            )

        return SegmentationResult(
            masks=masks,
            labels=labels,
            confidences=confidences,
            bounding_boxes=bounding_boxes,
            source_width=source_width,
            source_height=source_height,
        )

    def _send_request(self, url: str, body: dict) -> dict:
        """Send a POST request and return the JSON response."""
        import urllib.request
        import urllib.error

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise SegmentationClientError(f"HTTP {e.code}: {e.reason}") from e
        except (urllib.error.URLError, OSError) as e:
            raise SegmentationClientError(f"Connection failed: {e}") from e

    @staticmethod
    def _decode_mask(mask_dict: dict) -> np.ndarray:
        """Decode a mask dict to a 2-D boolean numpy array."""
        if mask_dict.get("encoding") != "png_base64":
            raise SegmentationClientError(
                f"Unsupported mask encoding: {mask_dict.get('encoding')}"
            )
        data = base64.b64decode(mask_dict["data"])
        img = Image.open(io.BytesIO(data)).convert("L")
        arr = np.array(img)
        return arr > 0  # threshold to bool
