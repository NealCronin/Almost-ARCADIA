from unittest.mock import Mock, patch

import pytest
import requests

from core.errors import InstructionError
from core.services.instruction_client import InstructionClient
from core.services.specs import ServiceSpec


@patch("core.services.instruction_client.requests.post")
def test_start_service(mock_post: Mock) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"host": "192.168.1.20", "port": 8081, "service_type": "llm", "scheme": "http"}
    mock_post.return_value = response
    endpoint = InstructionClient("192.168.1.20", 9000).start_service(
        ServiceSpec("llm", 8081, {"model_path": "model.gguf"})
    )
    assert endpoint.base_url == "http://192.168.1.20:8081"
    response.raise_for_status.assert_called_once()


@patch("core.services.instruction_client.time.sleep")
@patch(
    "core.services.instruction_client.requests.get",
    side_effect=[requests.ConnectionError("down"), Mock(status_code=200)],
)
def test_health_retries_transient_connection(mock_get: Mock, _sleep: Mock) -> None:
    assert InstructionClient("host", 9000, retries=1).health() is True
    assert mock_get.call_count == 2


@patch("core.services.instruction_client.requests.post")
def test_validation_error_is_not_retried(mock_post: Mock) -> None:
    response = Mock(status_code=422)
    response.json.return_value = {"detail": "invalid port"}
    response.raise_for_status.side_effect = requests.HTTPError("422")
    mock_post.return_value = response
    with pytest.raises(InstructionError):
        InstructionClient("host", 9000, retries=3).start_service(ServiceSpec("llm", 8081, {"model_path": "m"}))
    assert mock_post.call_count == 1
