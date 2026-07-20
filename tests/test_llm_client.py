from __future__ import annotations

from typing import Any

from core.inference.llm_client import LLMClient
from core.services.specs import ServiceEndpoint


class Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


def test_client_discovers_and_preserves_server_model_id(monkeypatch):
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def fake_get(url: str, **kwargs: Any) -> Response:
        calls.append(("GET", url, None))
        return Response({"data": [{"id": "Qwen3.5-0.8B-IQ4_NL.gguf"}]})

    def fake_post(url: str, *, json: dict[str, Any], **kwargs: Any) -> Response:
        calls.append(("POST", url, json))
        return Response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("core.inference.llm_client.requests.get", fake_get)
    monkeypatch.setattr("core.inference.llm_client.requests.post", fake_post)

    client = LLMClient(ServiceEndpoint("127.0.0.1", 8081, "llm"))
    assert client.chat("hello").text == "ok"
    assert calls[1][2]["model"] == "Qwen3.5-0.8B-IQ4_NL.gguf"  # type: ignore[index]

    # The discovered ID is cached and no second /v1/models call is required.
    client.chat("again")
    assert sum(1 for method, _url, _payload in calls if method == "GET") == 1
