from __future__ import annotations

from core.config import ConfiguredService, NodeConfig
from core.services.specs import ServiceSpec
from web.forms import LLMServiceForm, RemoteNodeForm, SAMServiceForm

NODES = {
    "local": NodeConfig("local", "127.0.0.1"),
    "desktop": NodeConfig("remote", "100.96.40.81", 9000),
}


def llm_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "node": "local",
        "inference_port": 8081,
        "hf_repo": "org/model-GGUF",
        "n_ctx": 32768,
        "vision_enabled": False,
        "temperature": 0.2,
        "top_k": 40,
        "min_p": 0.05,
        "top_p": 0.95,
        "local_bind_host": "127.0.0.1",
        "n_gpu_layers": "all",
        "n_batch": 2048,
        "n_ubatch": 512,
        "flash_attn": "on",
        "use_mmap": True,
    }
    data.update(overrides)
    return data


def test_remote_node_form_normalizes_safe_names_and_rejects_invalid_addresses() -> None:
    form = RemoteNodeForm({"name": "GPU Desktop", "host": "192.168.1.20", "instruction_port": 9000})
    assert form.is_valid(), form.errors
    assert form.cleaned_data["name"] == "gpu-desktop"


def test_llm_quick_fields_are_declared_in_required_order() -> None:
    form = LLMServiceForm(nodes=NODES)
    assert list(form.fields)[:8] == [
        "node",
        "inference_port",
        "hf_repo",
        "n_ctx",
        "vision_enabled",
        "mmproj_repo",
        "temperature",
        "top_k",
    ]
    assert not {"model_source", "model_path", "hf_file", "hf_cache_dir", "n_parallel"}.intersection(form.fields)


def test_llm_repository_and_generation_validation(monkeypatch) -> None:
    monkeypatch.setattr("core.services.llm_settings.local_ipv4_addresses", lambda: {"127.0.0.1", "10.0.0.3"})
    valid = LLMServiceForm(llm_data(hf_repo=" org/model-GGUF ", local_bind_host="10.0.0.3"), nodes=NODES)
    assert valid.is_valid(), valid.errors
    spec = valid.to_spec()
    assert spec.settings["hf_repo"] == "org/model-GGUF"
    assert spec.settings["bind_host"] == "10.0.0.3"
    assert {"temperature", "top_k", "min_p", "top_p"}.issubset(spec.settings)
    for repository in ("", "https://huggingface.co/a/b", "/tmp/model", "a/../b", "--x/y"):
        assert not LLMServiceForm(llm_data(hf_repo=repository), nodes=NODES).is_valid()
    assert not LLMServiceForm(llm_data(n_batch=1, n_ubatch=2), nodes=NODES).is_valid()
    assert not LLMServiceForm(llm_data(top_p=0), nodes=NODES).is_valid()


def test_remote_node_address_wins_over_posted_local_host() -> None:
    form = LLMServiceForm(llm_data(node="desktop", local_bind_host="127.0.0.99"), nodes=NODES)
    assert form.is_valid(), form.errors
    assert form.to_spec().settings["bind_host"] == "100.96.40.81"


def test_legacy_settings_migrate_without_losing_unknown_values() -> None:
    configured = ConfiguredService(
        "local",
        ServiceSpec("llm", 8081, {"hf_repo": "org/model", "hf_file": "model.gguf", "future": {"keep": True}}),
    )
    form = LLMServiceForm(nodes=NODES)
    form.initial_from(configured)
    assert form.initial["model_file_pattern"] == "model.gguf"
    bound = LLMServiceForm(llm_data(model_file_pattern="model.gguf"), nodes=NODES)
    bound.initial_from(configured)
    assert bound.is_valid(), bound.errors
    assert bound.to_spec().settings["future"] == {"keep": True}


def test_legacy_local_path_requires_repository_migration() -> None:
    form = LLMServiceForm(nodes=NODES)
    form.initial_from(ConfiguredService("local", ServiceSpec("llm", 8081, {"model_path": "/old/model.gguf"})))
    assert form.legacy_local_model
    assert not form.initial.get("hf_repo")


def test_sam_form_remains_available() -> None:
    form = SAMServiceForm(
        {
            "node": "local",
            "inference_port": 8090,
            "bind_host": "127.0.0.1",
            "checkpoint": "x.pt",
            "confidence": 0.25,
            "startup_timeout": 10,
        },
        nodes=NODES,
    )
    assert form.is_valid(), form.errors
