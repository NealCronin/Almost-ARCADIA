from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import requests

from core.services.specs import ServiceEndpoint


@dataclass(slots=True)
class SegmentationResult:
    masks: list[Any]
    labels: list[str]
    confidences: list[float]
    bounding_boxes: list[Any]


class SAMClient:
    def __init__(self, endpoint: ServiceEndpoint, timeout: float = 120.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def segment(
        self,
        frame: np.ndarray,
        prompts: list[str],
        confidence: float = 0.25,
    ) -> SegmentationResult:
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("Failed to encode frame.")

        image_base64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        response = requests.post(
            f"{self.endpoint.base_url}/v1/predict",
            json={
                "image": image_base64,
                "prompts": prompts,
                "confidence": confidence,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        return SegmentationResult(
            masks=payload.get("masks", []),
            labels=payload.get("labels", []),
            confidences=payload.get("confidences", payload.get("scores", [])),
            bounding_boxes=payload.get(
                "bounding_boxes",
                payload.get("boxes", payload.get("bbox", [])),
            ),
        )
