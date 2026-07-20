from __future__ import annotations

import pytest

from core.services.llm_runtime import LLMRuntime
from core.services.specs import ServiceSpec


def settings(**updates):
    values = {
        "hf_repo": "owner/repo",
        "hf_revision": "main",
        "hf_file": "model.gguf",
        "bind_host": "127.0.0.1",
        "n_ctx": 8192,
        "vision_enabled": False,
        "temperature": 0.1,
        "max_tokens": 256,
        "extra_args": ["--flash-attn", "on", "--mlock"],
    }
    values.update(updates)
    return values


def test_select_unique_unsplit_model():
    assert LLMRuntime._select_file(["README.md", "model.gguf"], None, projector=False) == "model.gguf"


def test_select_repo_requires_exact_when_ambiguous():
    with pytest.raises(ValueError, match="multiple usable"):
        LLMRuntime._select_file(["q4.gguf", "q8.gguf"], None, projector=False)


def test_split_selection_requires_first_shard():
    with pytest.raises(ValueError, match="first split shard"):
        LLMRuntime._select_file(["model-00002-of-00002.gguf"], "model-00002-of-00002.gguf", projector=False)


def test_build_command_has_only_quick_owned_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(LLMRuntime, "_find_executable", classmethod(lambda cls: "/bin/llama-server"))
    monkeypatch.setattr(LLMRuntime, "_resolve_model_path", classmethod(lambda cls, value: str(tmp_path / "model.gguf")))
    spec = ServiceSpec("llm", 8081, settings())
    command = LLMRuntime.build_command(spec)
    assert command[:3] == ["/bin/llama-server", "--host", "127.0.0.1"]
    assert "--model" in command
    assert "--ctx-size" in command
    assert "--alias" not in command
    assert "--temperature" not in command
    assert "--predict" not in command
    assert command[-3:] == ["on", "--mlock"] or command[-3:] == ["--flash-attn", "on", "--mlock"]


def test_build_command_adds_projector(monkeypatch, tmp_path):
    monkeypatch.setattr(LLMRuntime, "_find_executable", classmethod(lambda cls: "/bin/llama-server"))
    monkeypatch.setattr(LLMRuntime, "_resolve_model_path", classmethod(lambda cls, value: str(tmp_path / "model.gguf")))
    monkeypatch.setattr(
        LLMRuntime, "_resolve_projector_path", classmethod(lambda cls, value: str(tmp_path / "mmproj.gguf"))
    )
    spec = ServiceSpec(
        "visual_llm",
        8082,
        settings(
            vision_enabled=True,
            mmproj_repo="owner/projectors",
            mmproj_revision="main",
            mmproj_file="mmproj.gguf",
        ),
    )
    command = LLMRuntime.build_command(spec)
    assert command[command.index("--mmproj") + 1].endswith("mmproj.gguf")


def test_sam_endpoint_uses_configured_inference_ip():
    from core.services.sam_runtime import SAMRuntime

    spec = ServiceSpec("sam3", 8090, {"bind_host": "10.0.0.25"})
    assert SAMRuntime.endpoint(spec, "192.168.1.20").host == "10.0.0.25"


def test_user_can_set_native_alias_in_additional_arguments(monkeypatch, tmp_path):
    monkeypatch.setattr(LLMRuntime, "_find_executable", classmethod(lambda cls: "/bin/llama-server"))
    monkeypatch.setattr(
        LLMRuntime,
        "_resolve_model_path",
        classmethod(lambda cls, value: str(tmp_path / "model.gguf")),
    )
    spec = ServiceSpec("llm", 8081, settings(extra_args=["--alias", "my-native-alias"]))
    command = LLMRuntime.build_command(spec)
    assert command[-2:] == ["--alias", "my-native-alias"]


def test_hugging_face_cache_layout(monkeypatch, tmp_path):
    import core.services.llm_runtime as llm_runtime

    monkeypatch.delenv("ARCADIA_HUGGINGFACE_DIR", raising=False)
    monkeypatch.delenv("ARCADIA_MODELS_DIR", raising=False)
    monkeypatch.setattr(llm_runtime, "BASE_DIR", tmp_path)

    assert LLMRuntime.huggingface_directory() == tmp_path / "huggingface"
    assert LLMRuntime.cache_directory("models") == tmp_path / "huggingface" / "models"
    assert LLMRuntime.cache_directory("mmproj") == tmp_path / "huggingface" / "mmproj"
    assert (tmp_path / "huggingface" / "models").is_dir()
    assert (tmp_path / "huggingface" / "mmproj").is_dir()


def test_models_directory_override_is_hugging_face_root(monkeypatch, tmp_path):
    custom = tmp_path / "shared-hf"
    monkeypatch.setenv("ARCADIA_MODELS_DIR", str(custom))

    assert LLMRuntime.cache_directory("models") == custom / "models"
    assert LLMRuntime.cache_directory("mmproj") == custom / "mmproj"


def test_huggingface_override_precedes_deprecated_models_override(monkeypatch, tmp_path):
    preferred = tmp_path / "preferred"
    legacy = tmp_path / "legacy"
    monkeypatch.setenv("ARCADIA_HUGGINGFACE_DIR", str(preferred))
    monkeypatch.setenv("ARCADIA_MODELS_DIR", str(legacy))

    assert LLMRuntime.huggingface_directory() == preferred
