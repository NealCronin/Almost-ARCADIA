from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from core.services.sam_runtime import create_app


class FakePredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image, prompts, confidence):
        self.calls += 1
        return {
            "masks": [[[1, 0], [0, 1]]],
            "labels": [prompts[0]],
            "confidences": [confidence],
            "bounding_boxes": [[0, 0, 1, 1]],
        }


def _image_payload() -> str:
    ok, encoded = cv2.imencode(".jpg", np.zeros((3, 3, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def test_sam_endpoint_decodes_and_serializes() -> None:
    predictor = FakePredictor()
    client = TestClient(create_app("unused.pt", predictor=predictor))
    response = client.post("/v1/predict", json={"image": _image_payload(), "prompts": ["car"], "confidence": 0.4})
    assert response.status_code == 200
    assert response.json()["labels"] == ["car"]
    assert predictor.calls == 1


def test_sam_endpoint_rejects_empty_prompts() -> None:
    client = TestClient(create_app("unused.pt", predictor=FakePredictor()))
    response = client.post("/v1/predict", json={"image": _image_payload(), "prompts": []})
    assert response.status_code == 422
