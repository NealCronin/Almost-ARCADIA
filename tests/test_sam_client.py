from __future__ import annotations

import pytest

from core.errors import InferenceError
from core.inference.sam_client import SAMClient
from core.services.specs import ServiceEndpoint


class ErrorResponse:
    ok = False
    status_code = 422
    text = '{"detail":[{"msg":"Field required","loc":["body","text"]}]}'


def test_client_preserves_server_validation_detail(monkeypatch):
    def fake_post(_url, **_kwargs):
        return ErrorResponse()

    monkeypatch.setattr("core.inference.sam_client.requests.post", fake_post)
    client = SAMClient(ServiceEndpoint("127.0.0.1", 8090, "sam3"))

    with pytest.raises(InferenceError, match='HTTP 422: .*"text"'):
        client.segment(__import__("numpy").zeros((2, 2, 3), dtype="uint8"), ["person"])


def test_client_uses_canonical_request_for_each_priority_map_term(monkeypatch):
    calls = []

    class Response:
        ok = True

        @staticmethod
        def json():
            return {"detections": [], "overlay_png_base64": ""}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return Response()

    monkeypatch.setattr("core.inference.sam_client.requests.post", fake_post)
    client = SAMClient(ServiceEndpoint("127.0.0.1", 8090, "sam3"))

    result = client.segment(__import__("numpy").zeros((2, 2, 3), dtype="uint8"), ["car", "person", "car"])

    assert result.masks == []
    assert [payload["text"] for _url, payload in calls] == ["car", "person"]
    assert all(set(payload) == {"image_base64", "text", "confidence"} for _url, payload in calls)
