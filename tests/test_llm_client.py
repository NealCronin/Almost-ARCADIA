from unittest.mock import Mock, patch

from core.inference.llm_client import LLMClient
from core.services.specs import ServiceEndpoint


@patch("core.inference.llm_client.requests.post")
def test_chat_returns_message(mock_post: Mock) -> None:
    response = Mock()
    response.json.return_value = {
        "choices": [{"message": {"content": "hello"}}]
    }
    mock_post.return_value = response

    client = LLMClient(
        ServiceEndpoint(
            host="127.0.0.1",
            port=8081,
            service_type="llm",
        )
    )

    assert client.chat("Hi") == "hello"
