"""Tests for SegmentationClient.

Uses a mocked HTTP transport to test the client without a running server.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from arcadia.contracts import ServiceEndpoint, SegmentationRequest, SegmentationResult
from arcadia.inference import SegmentationClient, SegmentationClientError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_mask_png(width=50, height=50):
    """Create a valid PNG mask image and return base64 data."""
    img = Image.new("L", (width, height), color=0)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSegmentationClientInit:
    def test_stores_timeout(self):
        client = SegmentationClient(timeout=60.0)
        assert client._timeout == 60.0

    def test_stores_custom_urlopen(self):
        fake = lambda *a, **kw: None
        client = SegmentationClient(urlopen=fake)
        assert client._urlopen is fake


class TestSegmentEndpointValidation:
    def test_wrong_service_type_raises(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="generation")
        request = SegmentationRequest(image=b"fake", prompt="person")
        with pytest.raises(SegmentationClientError, match="service type must be 'segmentation'"):
            client.segment(endpoint, request)

    def test_empty_image_raises(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"", prompt="person")
        with pytest.raises(SegmentationClientError, match="Empty image bytes"):
            client.segment(endpoint, request)

    def test_empty_prompt_raises(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt=[])
        with pytest.raises(SegmentationClientError, match="Empty prompt"):
            client.segment(endpoint, request)


class TestSegmentHTTPTransport:
    def test_sends_correct_url(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = {"predictions": [], "source_width": 0, "source_height": 0}
            client.segment(endpoint, request)
            url = mock_send.call_args[0][0]
            assert url == "http://127.0.0.1:8080/v1/segment"

    def test_sends_base64_encoded_image(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        image_bytes = b"fake image data"
        request = SegmentationRequest(image=image_bytes, prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = {"predictions": [], "source_width": 0, "source_height": 0}
            client.segment(endpoint, request)
            body = mock_send.call_args[0][1]
            assert body["image_base64"] == base64.b64encode(image_bytes).decode("ascii")

    def test_sends_prompts_as_list(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = {"predictions": [], "source_width": 0, "source_height": 0}
            client.segment(endpoint, request)
            body = mock_send.call_args[0][1]
            assert body["prompts"] == ["person"]

    def test_sends_list_prompts_as_is(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt=["person", "dog"])

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = {"predictions": [], "source_width": 0, "source_height": 0}
            client.segment(endpoint, request)
            body = mock_send.call_args[0][1]
            assert body["prompts"] == ["person", "dog"]

    def test_includes_confidence_when_supplied(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = {"predictions": [], "source_width": 0, "source_height": 0}
            client.segment(endpoint, request, confidence=0.5)
            body = mock_send.call_args[0][1]
            assert body["confidence"] == 0.5

    def test_excludes_confidence_when_not_supplied(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = {"predictions": [], "source_width": 0, "source_height": 0}
            client.segment(endpoint, request)
            body = mock_send.call_args[0][1]
            assert "confidence" not in body


class TestSegmentResponseParsing:
    def test_valid_response_returns_result(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        fake_response = {
            "predictions": [
                {
                    "label": "person",
                    "confidence": 0.93,
                    "bounding_box": [10.0, 20.0, 90.0, 80.0],
                    "mask": {
                        "encoding": "png_base64",
                        "data": _make_valid_mask_png(),
                        "width": 50,
                        "height": 50,
                    },
                }
            ],
            "source_width": 100,
            "source_height": 100,
        }

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = fake_response
            result = client.segment(endpoint, request)

        assert isinstance(result, SegmentationResult)
        assert len(result.masks) == 1
        assert result.labels == ["person"]
        assert result.confidences == [0.93]
        assert result.bounding_boxes == [[10.0, 20.0, 90.0, 80.0]]
        assert result.source_width == 100
        assert result.source_height == 100

    def test_empty_predictions_returns_valid_result(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = {"predictions": [], "source_width": 0, "source_height": 0}
            result = client.segment(endpoint, request)

        assert isinstance(result, SegmentationResult)
        assert len(result.masks) == 0
        assert result.labels == []
        assert result.confidences == []
        assert result.bounding_boxes == []

    def test_mask_decodes_to_bool_array(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        fake_response = {
            "predictions": [
                {
                    "label": "person",
                    "confidence": 0.93,
                    "bounding_box": [10.0, 20.0, 90.0, 80.0],
                    "mask": {
                        "encoding": "png_base64",
                        "data": _make_valid_mask_png(),
                        "width": 50,
                        "height": 50,
                    },
                }
            ],
            "source_width": 100,
            "source_height": 100,
        }

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = fake_response
            result = client.segment(endpoint, request)

        assert len(result.masks) == 1
        mask = result.masks[0]
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (50, 50)

    def test_wrong_mask_encoding_raises(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        fake_response = {
            "predictions": [
                {
                    "label": "person",
                    "confidence": 0.93,
                    "bounding_box": [10.0, 20.0, 90.0, 80.0],
                    "mask": {
                        "encoding": "raw_bytes",
                        "data": "fake",
                        "width": 50,
                        "height": 50,
                    },
                }
            ],
            "source_width": 100,
            "source_height": 100,
        }

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = fake_response
            with pytest.raises(SegmentationClientError, match="Unsupported mask encoding"):
                client.segment(endpoint, request)

    def test_corrupt_mask_png_raises(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        fake_response = {
            "predictions": [
                {
                    "label": "person",
                    "confidence": 0.93,
                    "bounding_box": [10.0, 20.0, 90.0, 80.0],
                    "mask": {
                        "encoding": "png_base64",
                        "data": "not-a-valid-png-base64!!!",
                        "width": 50,
                        "height": 50,
                    },
                }
            ],
            "source_width": 100,
            "source_height": 100,
        }

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = fake_response
            with pytest.raises(SegmentationClientError):
                client.segment(endpoint, request)


class TestSegmentHTTPFailure:
    def test_http_error_raises(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.side_effect = SegmentationClientError("HTTP 500: Internal Server Error")
            with pytest.raises(SegmentationClientError, match="HTTP 500"):
                client.segment(endpoint, request)

    def test_connection_error_raises(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.side_effect = SegmentationClientError("Connection refused")
            with pytest.raises(SegmentationClientError, match="Connection refused"):
                client.segment(endpoint, request)

    def test_malformed_json_raises(self):
        client = SegmentationClient()
        endpoint = ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation")
        request = SegmentationRequest(image=b"fake", prompt="person")

        with mock.patch.object(client, "_send_request") as mock_send:
            mock_send.return_value = "not json"
            with pytest.raises(SegmentationClientError, match="Invalid response"):
                client.segment(endpoint, request)