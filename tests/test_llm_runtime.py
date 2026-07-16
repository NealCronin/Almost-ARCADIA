from unittest.mock import Mock, patch

import pytest

from core.errors import ServiceStartupError
from core.services.llm_runtime import LLMRuntime
from core.services.specs import ServiceSpec


def test_build_command_supports_huggingface_and_extra_args() -> None:
    command = LLMRuntime.build_command(
        ServiceSpec("llm", 8081, {"hf_repo": "org/model", "hf_file": "model.gguf", "extra_args": ["--verbose"]})
    )
    assert command[:3] == [command[0], "-m", "llama_cpp.server"]
    assert "--hf_repo" in command and "--hf_file" in command
    assert command[-1] == "--verbose"


def test_build_command_supports_local_model() -> None:
    command = LLMRuntime.build_command(ServiceSpec("llm", 8081, {"model_path": "model.gguf"}))
    assert "--model" in command and "model.gguf" in command


def test_command_escape_hatch_is_test_only() -> None:
    spec = ServiceSpec("llm", 8081, {"command": ["fake"]})
    with pytest.raises(ValueError):
        LLMRuntime.build_command(spec)
    assert LLMRuntime.build_command(spec, allow_test_command=True) == ["fake"]


@patch("core.services.llm_runtime.LLMRuntime.probe")
def test_wait_ready_polls_until_success(mock_probe: Mock) -> None:
    process = Mock()
    process.poll.return_value = None
    response = Mock(status_code=200)
    mock_probe.return_value = response
    from core.services.specs import ServiceEndpoint

    LLMRuntime.wait_ready(process, ServiceEndpoint("127.0.0.1", 8081, "llm"), timeout=1, poll_interval=0)
    mock_probe.assert_called_once()


def test_wait_ready_reports_dead_child() -> None:
    process = Mock()
    process.poll.return_value = 3
    process.returncode = 3
    from core.services.specs import ServiceEndpoint

    with pytest.raises(ServiceStartupError):
        LLMRuntime.wait_ready(process, ServiceEndpoint("127.0.0.1", 8081, "llm"), timeout=1, poll_interval=0)
