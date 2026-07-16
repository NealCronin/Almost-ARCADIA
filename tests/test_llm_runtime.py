from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.errors import ServiceStartupError
from core.services.llm_runtime import LLMRuntime
from core.services.specs import ServiceEndpoint, ServiceSpec


def test_build_command_resolves_repository_and_runtime_flags(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.Q4.gguf", "mmproj.gguf"])
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda repo, filename, cache: f"/{cache}/{filename}")
    command = LLMRuntime.build_command(
        ServiceSpec(
            "llm",
            8081,
            {
                "hf_repo": "org/model",
                "bind_host": "127.0.0.1",
                "n_ctx": 4096,
                "n_batch": 512,
                "vision_enabled": True,
                "model_file_pattern": "*Q4.gguf",
                "mmproj_file_pattern": "mmproj.gguf",
            },
        )
    )
    assert command[:8] == [command[0], "-m", "llama_cpp.server", "--host", "127.0.0.1", "--port", "8081", "--model"]
    assert "/huggingface/model.Q4.gguf" in command
    assert ["--clip_model_path", "/mmproj/mmproj.gguf"] == command[
        command.index("--clip_model_path") : command.index("--clip_model_path") + 2
    ]
    assert "--hf_model_repo_id" not in command and "--n_parallel" not in command


def test_ambiguous_and_split_models_require_safe_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        LLMRuntime, "list_repository_files", lambda _: ["a.Q4.gguf", "b.Q8.gguf", "split-00001-of-00002.gguf"]
    )
    with pytest.raises(ValueError, match="Model file pattern"):
        LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model"}))
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["split-00001-of-00002.gguf"])
    with pytest.raises(ValueError, match="No usable"):
        LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model"}))


def test_models_directory_honors_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARCADIA_MODELS_DIR", str(tmp_path / "models"))
    assert LLMRuntime.models_directory() == tmp_path / "models"


@patch("core.services.llm_runtime.hf_hub_download", create=True)
def test_download_uses_per_machine_cache(mock_download: Mock, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ARCADIA_MODELS_DIR", str(tmp_path / "models"))
    mock_download.return_value = tmp_path / "model.gguf"
    with patch.dict("sys.modules", {}):
        pass
    with patch("huggingface_hub.hf_hub_download", mock_download):
        LLMRuntime._download_hf_model("org/model", "model.gguf", "huggingface")
    assert (tmp_path / "models" / "huggingface").is_dir()
    assert mock_download.call_args.kwargs["token"] is None


@patch("core.services.llm_runtime.requests.get")
def test_repository_metadata_is_bounded_and_uses_hub_auth_headers(mock_get: Mock) -> None:
    response = Mock()
    response.json.return_value = [{"path": "model.gguf"}]
    mock_get.return_value = response
    assert LLMRuntime.list_repository_files("org/model") == ["model.gguf"]
    assert mock_get.call_args.kwargs["timeout"] == (5, 15)
    assert mock_get.call_args.kwargs["params"]["limit"] == 500


def test_legacy_local_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="retired"):
        LLMRuntime.build_command(ServiceSpec("llm", 8081, {"model_path": "old.gguf"}))


def test_readiness_targets_openai_models_route() -> None:
    assert LLMRuntime.readiness_url(ServiceEndpoint("127.0.0.1", 8081, "llm")) == "http://127.0.0.1:8081/v1/models"


def test_wait_ready_reports_dead_child() -> None:
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(ServiceStartupError):
        LLMRuntime.wait_ready(process, ServiceEndpoint("127.0.0.1", 8081, "llm"), timeout=1, poll_interval=0)
