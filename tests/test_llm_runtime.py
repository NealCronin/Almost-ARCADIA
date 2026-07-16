from unittest.mock import Mock, patch

import pytest

from core.errors import ServiceStartupError
from core.services.llm_runtime import LLMRuntime
from core.services.specs import ServiceEndpoint, ServiceSpec


def test_build_command_supports_local_model_and_pinned_server_flags() -> None:
    command = LLMRuntime.build_command(
        ServiceSpec(
            "llm",
            8081,
            {
                "model_path": "model.gguf",
                "n_ctx": 8192,
                "n_gpu_layers": 33,
                "chat_format": "chatml",
                "model_alias": "research-model",
                "extra_args": ["--verbose", "false"],
            },
        )
    )
    assert command[:3] == [command[0], "-m", "llama_cpp.server"]
    assert command[command.index("--model") + 1] == "model.gguf"
    assert command[command.index("--n_ctx") + 1] == "8192"
    assert command[command.index("--n_gpu_layers") + 1] == "33"
    assert command[command.index("--chat_format") + 1] == "chatml"
    assert command[command.index("--model_alias") + 1] == "research-model"
    assert command[-2:] == ["--verbose", "false"]


@patch("core.services.llm_runtime.LLMRuntime._download_hf_model", return_value="cache/model.gguf")
def test_build_command_resolves_exact_huggingface_file(mock_download: Mock) -> None:
    command = LLMRuntime.build_command(
        ServiceSpec("llm", 8081, {"hf_repo": "org/model", "hf_file": "model-q4.gguf", "hf_cache_dir": "cache"})
    )
    mock_download.assert_called_once_with("org/model", "model-q4.gguf", "cache")
    assert command[command.index("--model") + 1] == "cache/model.gguf"
    assert "--hf_repo" not in command
    assert "--hf_file" not in command


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"hf_repo": "org/model"},
        {"hf_file": "model.gguf"},
        {"model_path": "model.gguf", "hf_repo": "org/model", "hf_file": "model.gguf"},
    ],
)
def test_build_command_rejects_invalid_model_source_combinations(settings) -> None:
    with pytest.raises(ValueError):
        LLMRuntime.build_command(ServiceSpec("llm", 8081, settings))


def test_command_escape_hatch_is_test_only() -> None:
    spec = ServiceSpec("llm", 8081, {"command": ["fake"]})
    with pytest.raises(ValueError):
        LLMRuntime.build_command(spec)
    assert LLMRuntime.build_command(spec, allow_test_command=True) == ["fake"]


def test_readiness_targets_openai_models_route() -> None:
    assert LLMRuntime.readiness_url(ServiceEndpoint("127.0.0.1", 8081, "llm")) == "http://127.0.0.1:8081/v1/models"


@patch("core.services.llm_runtime.LLMRuntime.probe")
def test_wait_ready_polls_until_success(mock_probe: Mock) -> None:
    process = Mock()
    process.poll.return_value = None
    mock_probe.return_value = Mock(status_code=200)
    LLMRuntime.wait_ready(process, ServiceEndpoint("127.0.0.1", 8081, "llm"), timeout=1, poll_interval=0)
    mock_probe.assert_called_once()


def test_wait_ready_reports_dead_child() -> None:
    process = Mock()
    process.poll.return_value = 3
    process.returncode = 3
    with pytest.raises(ServiceStartupError):
        LLMRuntime.wait_ready(process, ServiceEndpoint("127.0.0.1", 8081, "llm"), timeout=1, poll_interval=0)
