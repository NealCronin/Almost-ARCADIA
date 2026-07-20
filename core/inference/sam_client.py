from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
import requests

from core.errors import InferenceError
from core.inference.results import SegmentationResult
from core.services.specs import ServiceEndpoint


class SAMClient:
    def __init__(self, endpoint: ServiceEndpoint, timeout: float = 180.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def segment(
        self,
        frame: np.ndarray,
        prompts: list[str],
        *,
        confidence: float = 0.25,
        resize: tuple[int, int] | None = None,
    ) -> SegmentationResult:
        if not 0 <= confidence <= 1:
            raise InferenceError("SAM3 confidence must be between 0.0 and 1.0.", service_type="sam3")
        search_terms = list(
            dict.fromkeys(prompt.strip() for prompt in prompts if isinstance(prompt, str) and prompt.strip())
        )
        if not search_terms:
            raise InferenceError("SAM3 requires at least one non-empty search term.", service_type="sam3")
        image = frame
        if resize is not None:
            image = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise InferenceError("Could not encode a frame for SAM3.", service_type="sam3")
        image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        masks: list[Any] = []
        labels: list[str] = []
        confidences: list[float] = []
        bounding_boxes: list[Any] = []
        try:
            for text in search_terms:
                response = requests.post(
                    f"{self.endpoint.base_url}/v1/predict",
                    json={"image_base64": image_base64, "text": text, "confidence": confidence},
                    timeout=self.timeout,
                )
                if not response.ok:
                    raise InferenceError(
                        f"SAM3 inference failed at {self.endpoint.base_url}: "
                        f"HTTP {response.status_code}: {response.text}",
                        service_type="sam3",
                    )
                payload: dict[str, Any] = response.json()
                detections = payload.get("detections")
                if not isinstance(detections, list):
                    raise ValueError("SAM3 response did not contain a detections list.")
                for detection in detections:
                    if not isinstance(detection, dict):
                        continue
                    encoded_mask = detection.get("mask_png_base64")
                    if not isinstance(encoded_mask, str):
                        continue
                    raw_mask = base64.b64decode(encoded_mask, validate=True)
                    mask = cv2.imdecode(np.frombuffer(raw_mask, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        raise ValueError("SAM3 response contained an invalid mask PNG.")
                    masks.append((mask > 0).astype(np.uint8))
                    labels.append(str(detection.get("label", text)))
                    confidences.append(float(detection.get("confidence", confidence)))
                    box = detection.get("box")
                    bounding_boxes.append(box if isinstance(box, list) else [])
        except InferenceError:
            raise
        except (requests.RequestException, TypeError, ValueError) as exc:
            raise InferenceError(
                f"SAM3 inference failed at {self.endpoint.base_url}: {exc}", service_type="sam3"
            ) from exc
        return SegmentationResult(
            masks=masks,
            labels=labels,
            confidences=confidences,
            bounding_boxes=bounding_boxes,
        )
