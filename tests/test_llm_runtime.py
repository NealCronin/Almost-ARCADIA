from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.errors import ServiceStartupError
from core.services.llm_runtime import LLMRuntime
from core.services.specs import ServiceEndpoint, ServiceSpec


def test_build_command_resolves_repository_and_runtime_flags(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.Q4.gguf", "mmproj.gguf"])
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda repo, filename, cache: f"/{cache}/{filename}")
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/usr/local/bin/llama-server")
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
    assert command[:8] == [
        "/usr/local/bin/llama-server",
        "--host",
        "127.0.0.1",
        "--port",
        "8081",
        "--model",
        "/huggingface/model.Q4.gguf",
        "--mmproj",
    ]
    assert "/mmproj/mmproj.gguf" in command
    assert "--ctx-size" in command and "4096" in command


def test_ambiguous_and_split_models_require_safe_selection(monkeypatch) -> None:
    monkeypatch.setattr(
        LLMRuntime, "list_repository_files", lambda _: ["a.Q4.gguf", "b.Q8.gguf", "split-00001-of-00002.gguf"]
    )
    with pytest.raises(ValueError, match="select one"):
        LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model"}))
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["split-00001-of-00002.gguf"])
    with pytest.raises(ValueError, match="Only split GGUF"):
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


def test_readiness_targets_health_endpoint() -> None:
    assert LLMRuntime.readiness_url(ServiceEndpoint("127.0.0.1", 8081, "llm")) == "http://127.0.0.1:8081/health"


def test_wait_ready_reports_dead_child() -> None:
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(ServiceStartupError):
        LLMRuntime.wait_ready(process, ServiceEndpoint("127.0.0.1", 8081, "llm"), timeout=1, poll_interval=0)


# ── Flash Attention tri-state ──────────────────────────────────────────


def test_flash_attn_auto_emits_flag(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.gguf"])
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda *a: "/models/model.gguf")
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    command = LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model", "flash_attn": "auto"}))
    assert "--flash-attn" in command
    idx = command.index("--flash-attn")
    assert command[idx + 1] == "auto"


def test_flash_attn_on_emits_flag(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.gguf"])
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda *a: "/models/model.gguf")
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    command = LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model", "flash_attn": "on"}))
    assert "--flash-attn" in command
    idx = command.index("--flash-attn")
    assert command[idx + 1] == "on"


def test_flash_attn_off_emits_flag(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.gguf"])
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda *a: "/models/model.gguf")
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    command = LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model", "flash_attn": "off"}))
    assert "--flash-attn" in command
    idx = command.index("--flash-attn")
    assert command[idx + 1] == "off"


def test_flash_attn_no_numeric_one(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.gguf"])
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda *a: "/models/model.gguf")
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    command = LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model", "flash_attn": "on"}))
    assert "1" not in [command[i + 1] for i, a in enumerate(command) if a == "--flash-attn"]


def test_flash_attn_not_set_omits_flag(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.gguf"])
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda *a: "/models/model.gguf")
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    command = LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model"}))
    assert "--flash-attn" not in command


# ── Nested paths ────────────────────────────────────────────────────────


def test_nested_single_model_preserves_path(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["Q4_K/model.gguf"])
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    downloads = []
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda r, f, c: downloads.append(f) or f"/cache/{f}")
    LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model", "model_file_pattern": "*.gguf"}))
    assert downloads[0] == "Q4_K/model.gguf"


def test_nested_split_main_model(monkeypatch) -> None:
    files = [
        "Q6/model-00001-of-00003.gguf",
        "Q6/model-00002-of-00003.gguf",
        "Q6/model-00003-of-00003.gguf",
    ]
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: list(files))
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    downloads = []
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda r, f, c: downloads.append(f) or f"/cache/{f}")
    LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model", "model_file_pattern": "model-00001*"}))
    assert len(downloads) == 3
    assert "Q6/model-00001-of-00003.gguf" in downloads
    assert "Q6/model-00002-of-00003.gguf" in downloads
    assert "Q6/model-00003-of-00003.gguf" in downloads


def test_nested_split_draft_model(monkeypatch) -> None:
    main_files = ["main.gguf"]
    draft_files = [
        "draft/draft-00001-of-00002.gguf",
        "draft/draft-00002-of-00002.gguf",
    ]
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    downloads = []

    def _list_files(repo):
        return main_files if repo == "org/model" else draft_files

    monkeypatch.setattr(LLMRuntime, "list_repository_files", _list_files)
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda r, f, c: downloads.append(f) or f"/cache/{f}")
    LLMRuntime.build_command(
        ServiceSpec(
            "llm",
            8081,
            {
                "hf_repo": "org/model",
                "draft_enabled": True,
                "draft_repo": "org/draft-model",
                "draft_file_pattern": "draft-00001*",
            },
        )
    )
    assert "draft/draft-00001-of-00002.gguf" in downloads
    assert "draft/draft-00002-of-00002.gguf" in downloads


def test_missing_shard_reports_error(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model-00001-of-00003.gguf"])
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda r, f, c: f"/cache/{f}")
    with pytest.raises(ValueError, match="missing shard"):
        LLMRuntime.build_command(
            ServiceSpec("llm", 8081, {"hf_repo": "org/model", "model_file_pattern": "model-00001*"})
        )


def test_first_shard_enforcement(monkeypatch) -> None:
    monkeypatch.setattr(
        LLMRuntime,
        "list_repository_files",
        lambda _: ["model-00002-of-00003.gguf", "model-00003-of-00003.gguf"],
    )
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda r, f, c: f"/cache/{f}")
    with pytest.raises(ValueError, match="must start from shard 1"):
        LLMRuntime.build_command(
            ServiceSpec("llm", 8081, {"hf_repo": "org/model", "model_file_pattern": "model-00002*"})
        )


def test_nested_projector_path(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.gguf", "mmproj/mmproj.gguf"])
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    downloads = []
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda r, f, c: downloads.append(f) or f"/cache/{f}")
    LLMRuntime.build_command(
        ServiceSpec(
            "llm",
            8081,
            {
                "hf_repo": "org/model",
                "vision_enabled": True,
                "model_file_pattern": "model.gguf",
                "mmproj_file_pattern": "mmproj.gguf",
            },
        )
    )
    assert "mmproj/mmproj.gguf" in downloads


# ── Case-insensitive pattern ───────────────────────────────────────────


def test_case_insensitive_pattern(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["MODEL-Q6_K.gguf"])
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    downloads = []
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda r, f, c: downloads.append(f) or f"/cache/{f}")
    LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model", "model_file_pattern": "*q6_k*.gguf"}))
    assert downloads[0] == "MODEL-Q6_K.gguf"


# ── Projector excluded from main model ─────────────────────────────────


def test_projector_not_selected_as_main_model(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.gguf", "mmproj.gguf"])
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    downloads = []
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda r, f, c: downloads.append(f) or f"/cache/{f}")
    LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model"}))
    assert downloads[0] == "model.gguf"


# ── Command is always an argument list ──────────────────────────────────


def test_command_remains_argument_list(monkeypatch) -> None:
    monkeypatch.setattr(LLMRuntime, "list_repository_files", lambda _: ["model.gguf"])
    monkeypatch.setattr(LLMRuntime, "_download_hf_model", lambda *a: "/models/model.gguf")
    monkeypatch.setattr(LLMRuntime, "_find_executable", lambda: "/bin/llama-server")
    command = LLMRuntime.build_command(ServiceSpec("llm", 8081, {"hf_repo": "org/model", "flash_attn": "auto"}))
    assert isinstance(command, list)
    assert all(isinstance(arg, str) for arg in command)
