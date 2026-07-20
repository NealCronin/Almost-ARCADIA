from __future__ import annotations

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from core.services.sam_runtime import create_app


class FakePredictor:
    def __init__(self, detections):
        self.detections = detections
        self.calls: list[tuple[str, float]] = []

    def predict(self, image, text: str, confidence: float):
        self.calls.append((text, confidence))
        return self.detections


def encoded_image() -> str:
    image = np.full((4, 5, 3), 120, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def client_with(detections):
    predictor = FakePredictor(detections)
    return TestClient(create_app("unused.pt", predictor=predictor)), predictor


def test_valid_prediction_returns_compact_png_masks():
    client, predictor = client_with(
        [{"label": "person", "confidence": 0.93, "box": [0, 0, 4, 3], "mask": np.ones((4, 5), dtype=np.uint8)}]
    )

    response = client.post("/v1/predict", json={"image_base64": encoded_image(), "text": "person", "confidence": 0.25})

    assert response.status_code == 200
    payload = response.json()
    assert predictor.calls == [("person", 0.25)]
    assert payload["detections"][0]["label"] == "person"
    assert payload["detections"][0]["box"] == [0, 0, 4, 3]
    assert cv2.imdecode(
        np.frombuffer(base64.b64decode(payload["detections"][0]["mask_png_base64"]), dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE,
    ).shape == (4, 5)
    assert base64.b64decode(payload["overlay_png_base64"])


def test_predict_rejects_missing_search_text_with_fastapi_detail():
    client, _predictor = client_with([])

    response = client.post("/v1/predict", json={"image_base64": encoded_image(), "confidence": 0.25})

    assert response.status_code == 422
    assert response.json()["detail"]


def test_predict_rejects_invalid_confidence_with_fastapi_detail():
    client, _predictor = client_with([])

    response = client.post("/v1/predict", json={"image_base64": encoded_image(), "text": "person", "confidence": 1.1})

    assert response.status_code == 422
    assert response.json()["detail"]


def test_predict_rejects_invalid_base64():
    client, _predictor = client_with([])

    response = client.post("/v1/predict", json={"image_base64": "not base64", "text": "person"})

    assert response.status_code == 400
    assert "base64" in response.json()["detail"]


def test_predict_rejects_non_image_bytes():
    client, _predictor = client_with([])

    response = client.post(
        "/v1/predict",
        json={"image_base64": base64.b64encode(b"not an image").decode("ascii"), "text": "person"},
    )

    assert response.status_code == 400
    assert "decoded" in response.json()["detail"]


def test_predict_handles_empty_detections():
    client, _predictor = client_with([])

    response = client.post("/v1/predict", json={"image_base64": encoded_image(), "text": "person"})

    assert response.status_code == 200
    assert response.json()["detections"] == []


def test_predict_handles_multiple_masks_and_missing_boxes():
    client, _predictor = client_with(
        [
            {"label": "person", "confidence": 0.8, "box": None, "mask": np.ones((4, 5), dtype=np.uint8)},
            {"label": "person", "confidence": 0.7, "box": [1, 1, 3, 3], "mask": np.eye(4, 5, dtype=np.uint8)},
        ]
    )

    response = client.post("/v1/predict", json={"image_base64": encoded_image(), "text": "person"})

    assert response.status_code == 200
    detections = response.json()["detections"]
    assert len(detections) == 2
    assert detections[0]["box"] is None
    assert all(item["mask_png_base64"] for item in detections)


def test_predict_accepts_legacy_aliases_and_normalizes_them():
    client, predictor = client_with([])

    response = client.post("/v1/predict", json={"image": encoded_image(), "prompts": ["person"], "confidence": 0.25})

    assert response.status_code == 200
    assert predictor.calls == [("person", 0.25)]
