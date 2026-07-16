from __future__ import annotations

import base64
import threading
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.errors import ServiceStartupError
from core.services.sam_runtime import SAMRuntime, create_app
from core.services.specs import ServiceEndpoint


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


@patch("core.services.sam_runtime.SAMRuntime.probe")
def test_sam_readiness_wakes_when_cancelled_between_probes(mock_probe: Mock) -> None:
    process = Mock()
    process.poll.return_value = None
    cancelled = threading.Event()

    def non_ready(*_, **__):
        cancelled.set()
        return Mock(status_code=503)

    mock_probe.side_effect = non_ready
    with pytest.raises(ServiceStartupError, match="SAM3 startup cancelled"):
        SAMRuntime.wait_ready(
            process,
            ServiceEndpoint("127.0.0.1", 8090, "sam3"),
            timeout=30,
            poll_interval=30,
            cancel_event=cancelled,
        )

    mock_probe.assert_called_once()
