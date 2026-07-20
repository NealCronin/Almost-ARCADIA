from __future__ import annotations

from core.config import ConfiguredService, NodeConfig
from core.services.specs import ServiceSpec
from web.forms import LLMServiceForm, VisualLLMServiceForm

NODES = {
    "local": NodeConfig("local", "127.0.0.1"),
    "gpu": NodeConfig("remote", "192.168.1.20", 9000),
}


def llm_data(**updates):
    data = {
        "node": "gpu",
        "inference_port": "8081",
        "bind_host": "192.168.1.20",
        "hf_source": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/main/Qwen3.5-2B-IQ4_XS.gguf",
        "vision_enabled": "",
        "mmproj_source": "",
        "n_ctx": "32768",
        "max_tokens": "1024",
        "temperature": "0.1",
        "additional_arguments": "-ngl all, -fa on\n--mlock",
    }
    data.update(updates)
    return data


def test_llm_form_builds_simplified_spec():
    form = LLMServiceForm(llm_data(), nodes=NODES)
    assert form.is_valid(), form.errors
    spec = form.to_spec()
    assert spec.service_type == "llm"
    assert spec.settings["hf_file"] == "Qwen3.5-2B-IQ4_XS.gguf"
    assert spec.settings["bind_host"] == "192.168.1.20"
    assert spec.settings["extra_args"] == ["-ngl", "all", "-fa", "on", "--mlock"]
    assert "model_alias" not in spec.settings


def test_visual_form_forces_vision_and_requires_projector():
    form = VisualLLMServiceForm(llm_data(inference_port="8082"), nodes=NODES)
    assert not form.is_valid()
    assert "mmproj_source" in form.errors


def test_visual_form_uses_visual_service_type():
    form = VisualLLMServiceForm(
        llm_data(
            inference_port="8082",
            mmproj_source="https://huggingface.co/owner/projectors/blob/main/mmproj-model-f16.gguf",
        ),
        nodes=NODES,
    )
    assert form.is_valid(), form.errors
    spec = form.to_spec()
    assert spec.service_type == "visual_llm"
    assert spec.settings["vision_enabled"] is True
    assert "model_alias" not in spec.settings


def test_initial_from_round_trips_sources_and_arguments():
    configured = ConfiguredService(
        "gpu",
        ServiceSpec(
            "llm",
            8181,
            {
                "hf_repo": "owner/repo",
                "hf_revision": "dev",
                "hf_file": "folder/model.gguf",
                "bind_host": "10.0.0.5",
                "n_ctx": 4096,
                "vision_enabled": False,
                "temperature": 0.2,
                "max_tokens": 500,
                "model_alias": "logical-model",
                "extra_args": ["--mlock"],
                "models_cache_subdir": "huggingface",
            },
        ),
    )
    form = LLMServiceForm(nodes=NODES)
    form.initial_from(configured)
    assert form.initial["hf_source"].endswith("/blob/dev/folder/model.gguf")
    assert form.initial["additional_arguments"] == "--mlock"


def test_legacy_cache_subdir_is_not_persisted():
    form = LLMServiceForm(llm_data(), nodes=NODES)
    assert form.is_valid(), form.errors
    assert "models_cache_subdir" not in form.to_spec().settings
