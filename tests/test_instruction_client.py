from unittest.mock import Mock, patch

from core.services.instruction_client import InstructionClient
from core.services.specs import ServiceSpec


@patch("core.services.instruction_client.requests.post")
def test_start_service(mock_post: Mock) -> None:
    response = Mock()
    response.json.return_value = {
        "host": "192.168.1.20",
        "port": 8081,
        "service_type": "llm",
        "scheme": "http",
    }
    mock_post.return_value = response

    endpoint = InstructionClient("192.168.1.20", 9000).start_service(
        ServiceSpec(
            service_type="llm",
            port=8081,
            settings={"command": ["fake"]},
        )
    )

    assert endpoint.base_url == "http://192.168.1.20:8081"
    response.raise_for_status.assert_called_once()
