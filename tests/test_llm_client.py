from unittest.mock import Mock, patch

import pytest

from core.errors import InferenceError
from core.inference.llm_client import LLMClient
from core.services.specs import ServiceEndpoint


@patch("core.inference.llm_client.requests.post")
def test_chat_returns_typed_message(mock_post: Mock) -> None:
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "hello"}}]}
    mock_post.return_value = response
    result = LLMClient(ServiceEndpoint("127.0.0.1", 8081, "llm")).chat("Hi")
    assert result.text == "hello"
    assert mock_post.call_args.kwargs["json"]["messages"][0]["content"][0]["text"] == "Hi"


@patch("core.inference.llm_client.requests.post")
def test_chat_encodes_image_bytes(mock_post: Mock) -> None:
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_post.return_value = response
    LLMClient(ServiceEndpoint("127.0.0.1", 8081, "llm")).chat("describe", images=[("image/png", b"abc")])
    image_item = mock_post.call_args.kwargs["json"]["messages"][0]["content"][1]
    assert image_item["image_url"]["url"].startswith("data:image/png;base64,")


@patch("core.inference.llm_client.requests.post")
def test_chat_rejects_malformed_response(mock_post: Mock) -> None:
    response = Mock()
    response.json.return_value = {"choices": []}
    mock_post.return_value = response
    with pytest.raises(InferenceError) as exc_info:
        LLMClient(ServiceEndpoint("127.0.0.1", 8081, "llm")).chat("Hi")
    assert exc_info.value.service_type == "llm"


@patch("core.inference.llm_client.requests.post")
def test_chat_sends_generation_controls(mock_post: Mock) -> None:
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    mock_post.return_value = response
    LLMClient(ServiceEndpoint("127.0.0.1", 8081, "llm")).chat("Hi", temperature=0.3, top_k=12, min_p=0.1, top_p=0.8)
    body = mock_post.call_args.kwargs["json"]
    assert {key: body[key] for key in ("temperature", "top_k", "min_p", "top_p")} == {
        "temperature": 0.3,
        "top_k": 12,
        "min_p": 0.1,
        "top_p": 0.8,
    }
