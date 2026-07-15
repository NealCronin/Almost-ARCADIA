"""Tests for the SAM 3 segmentation HTTP server.

Uses a fake predictor to avoid loading the real model.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from arcadia.contracts import ServiceEndpoint
from arcadia.runtimes.sam_server import (
    app,
    state,
    _SAM3Predictor,
    _encode_mask,
)


# ---------------------------------------------------------------------------
# Fake predictor
# ---------------------------------------------------------------------------

class FakePredictor:
    """A fake predictor that returns configurable results."""

    def __init__(self, results: list[dict] | None = None):
        self.results = results or []

    def predict(self, image, prompts, confidence):
        if not prompts:
            raise ValueError("no prompts")
        return self.results


class FakePredictorWithException:
    """A fake predictor that raises an exception."""

    def predict(self, image, prompts, confidence):
        raise RuntimeError("model error")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_server_state(results=None, predictor=None):
    """Create a test server state with a fake predictor."""
    if predictor is not None:
        state.predictor = predictor
    else:
        state.predictor = FakePredictor(results=results)
    state.checkpoint_path = Path("/tmp/fake.pt")
    state.default_confidence = 0.25
    state.ready = True


def _make_test_image(width=100, height=100):
    """Create a small RGB image and return its base64 encoding."""
    img = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_state():
    """Reset server state before each test."""
    state.predictor = None
    state.ready = False
    yield
    state.predictor = None
    state.ready = False


class TestHealthEndpoint:
    def test_returns_loading_when_not_ready(self):
        _make_fake_server_state()
        state.ready = False
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "loading"
            assert data["service_type"] == "segmentation"

    def test_returns_ready_when_ready(self):
        _make_fake_server_state()
        with TestClient(app) as client:
            resp = client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ready"
            assert data["service_type"] == "segmentation"


class TestSegmentEndpoint:
    def test_valid_request_returns_predictions(self):
        _make_fake_server_state(results=[
            {
                "label": "person",
                "confidence": 0.93,
                "bounding_box": [10.0, 20.0, 90.0, 80.0],
                "mask": np.ones((100, 100), dtype=bool),
            }
        ])
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["predictions"]) == 1
            assert data["predictions"][0]["label"] == "person"
            assert data["predictions"][0]["confidence"] == 0.93
            assert data["predictions"][0]["bounding_box"] == [10.0, 20.0, 90.0, 80.0]
            assert data["source_width"] == 100
            assert data["source_height"] == 100

    def test_multiple_prompts_returns_multiple_predictions(self):
        _make_fake_server_state(results=[
            {"label": "person", "confidence": 0.9, "bounding_box": [0, 0, 50, 50], "mask": np.zeros((50, 50), dtype=bool)},
            {"label": "dog", "confidence": 0.8, "bounding_box": [50, 0, 100, 50], "mask": np.zeros((50, 50), dtype=bool)},
        ])
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person", "dog"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["predictions"]) == 2

    def test_no_match_returns_empty_predictions(self):
        _make_fake_server_state(results=[])
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["nonexistent"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["predictions"] == []

    def test_confidence_override(self):
        _make_fake_server_state(results=[
            {"label": "person", "confidence": 0.9, "bounding_box": [0, 0, 50, 50], "mask": np.zeros((50, 50), dtype=bool)},
        ])
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person"], "confidence": 0.5},
            )
            assert resp.status_code == 200

    def test_mask_encoding_is_png_base64(self):
        _make_fake_server_state(results=[
            {"label": "person", "confidence": 0.9, "bounding_box": [0, 0, 50, 50], "mask": np.zeros((50, 50), dtype=bool)},
        ])
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            mask = data["predictions"][0]["mask"]
            assert mask["encoding"] == "png_base64"
            assert "data" in mask
            assert "width" in mask
            assert "height" in mask

    def test_mask_decodes_to_bool_array(self):
        _make_fake_server_state(results=[
            {"label": "person", "confidence": 0.9, "bounding_box": [0, 0, 50, 50], "mask": np.zeros((50, 50), dtype=bool)},
        ])
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            mask_b64 = data["predictions"][0]["mask"]["data"]
            decoded = base64.b64decode(mask_b64)
            img = Image.open(io.BytesIO(decoded)).convert("L")
            arr = np.array(img)
            assert arr.shape == (50, 50)

    def test_invalid_base64_returns_400(self):
        _make_fake_server_state()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": "not-valid-base64!!!", "prompts": ["person"]},
            )
            assert resp.status_code == 400

    def test_unsupported_image_format_returns_400(self):
        _make_fake_server_state()
        image_b64 = base64.b64encode(b"not an image").decode("ascii")
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person"]},
            )
            assert resp.status_code == 400

    def test_empty_prompts_returns_400(self):
        _make_fake_server_state()
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": []},
            )
            assert resp.status_code == 422  # Pydantic validation

    def test_predictor_exception_returns_500(self):
        _make_fake_server_state(predictor=FakePredictorWithException())
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person"]},
            )
            assert resp.status_code == 500

    def test_label_confidence_box_mask_alignment(self):
        _make_fake_server_state(results=[
            {"label": "person", "confidence": 0.9, "bounding_box": [10.0, 20.0, 90.0, 80.0], "mask": np.zeros((100, 100), dtype=bool)},
            {"label": "dog", "confidence": 0.8, "bounding_box": [5.0, 15.0, 85.0, 75.0], "mask": np.zeros((100, 100), dtype=bool)},
        ])
        image_b64 = _make_test_image()
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person", "dog"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            preds = data["predictions"]
            assert len(preds) == 2
            assert preds[0]["label"] == "person"
            assert preds[1]["label"] == "dog"
            assert preds[0]["confidence"] == 0.9
            assert preds[1]["confidence"] == 0.8

    def test_source_dimensions_match_input(self):
        _make_fake_server_state(results=[
            {"label": "person", "confidence": 0.9, "bounding_box": [0, 0, 200, 300], "mask": np.zeros((300, 200), dtype=bool)},
        ])
        img = Image.new("RGB", (200, 300), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        with TestClient(app) as client:
            resp = client.post(
                "/v1/segment",
                json={"image_base64": image_b64, "prompts": ["person"]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["source_width"] == 200
            assert data["source_height"] == 300
