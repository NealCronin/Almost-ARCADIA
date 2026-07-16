from __future__ import annotations

import base64
from typing import Sequence

import cv2
import numpy as np
import requests

from core.errors import InferenceError
from core.inference.results import SegmentationResult
from core.services.specs import ServiceEndpoint


class SAMClient:
    """Call an already-running serialized SAM3 HTTP service."""

    def __init__(self, endpoint: ServiceEndpoint, timeout: float = 120.0) -> None:
        if endpoint.service_type != "sam3":
            raise ValueError("SAMClient requires a sam3 endpoint.")
        self.endpoint = endpoint
        self.timeout = timeout

    def segment(
        self,
        frame: np.ndarray,
        prompts: Sequence[str],
        confidence: float = 0.25,
        *,
        resize: tuple[int, int] | None = None,
    ) -> SegmentationResult:
        normalized_prompts = [str(prompt).strip() for prompt in prompts if str(prompt).strip()]
        if not normalized_prompts:
            raise ValueError("at least one SAM prompt is required")
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        image = frame
        if resize is not None:
            width, height = resize
            if width < 1 or height < 1:
                raise ValueError("SAM resize dimensions must be positive")
            image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise InferenceError("Failed to encode frame for SAM inference.")
        image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        try:
            response = requests.post(
                f"{self.endpoint.base_url}/v1/predict",
                json={"image": image_base64, "prompts": normalized_prompts, "confidence": confidence},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise InferenceError(f"SAM request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise InferenceError("SAM response must be a JSON object.")
        try:
            confidences = [float(value) for value in (payload.get("confidences") or payload.get("scores") or [])]
            labels = [str(value) for value in (payload.get("labels") or [])]
            masks = list(payload.get("masks") or [])
            boxes = list(payload.get("bounding_boxes") or payload.get("boxes") or payload.get("bbox") or [])
        except (TypeError, ValueError) as exc:
            raise InferenceError("SAM response contains invalid result arrays.") from exc
        return SegmentationResult(
            masks=masks,
            labels=labels,
            confidences=confidences,
            bounding_boxes=boxes,
        )
