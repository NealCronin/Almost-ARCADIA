from __future__ import annotations

from core.config import ConfiguredService, NodeConfig
from core.services.specs import ServiceSpec
from web.forms import LLMServiceForm, RemoteNodeForm, SAMServiceForm

NODES = {
    "local": NodeConfig("local", "127.0.0.1"),
    "desktop": NodeConfig("remote", "100.96.83.10", 9000),
}


def llm_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "node": "local",
        "inference_port": 8081,
        "bind_host": "0.0.0.0",
        "startup_timeout": 600,
        "model_source": "local",
        "model_path": "/models/model.gguf",
        "hf_repo": "",
        "hf_file": "",
        "hf_cache_dir": "",
        "n_ctx": 32768,
        "n_gpu_layers": -1,
        "n_threads": "",
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_parallel": 1,
        "flash_attn": "on",
        "cache_type_k": "",
        "cache_type_v": "",
        "chat_format": "",
        "model_alias": "local-model",
        "additional_arguments": "",
    }
    data.update(overrides)
    return data


def sam_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "node": "local",
        "inference_port": 8090,
        "bind_host": "0.0.0.0",
        "startup_timeout": 600,
        "checkpoint": "/models/sam3.pt",
        "confidence": 0.25,
        "additional_arguments": "",
    }
    data.update(overrides)
    return data


def test_remote_node_form_normalizes_safe_names_and_rejects_invalid_addresses() -> None:
    form = RemoteNodeForm({"name": "GPU Desktop", "host": "192.168.1.20", "instruction_port": 9000})

    assert form.is_valid(), form.errors
    assert form.cleaned_data["name"] == "gpu-desktop"
    assert form.to_config() == NodeConfig("remote", "192.168.1.20", 9000)

    invalid = RemoteNodeForm({"name": "../local", "host": "example.test", "instruction_port": "true"})
    assert not invalid.is_valid()
    assert "name" in invalid.errors
    assert "host" in invalid.errors
    assert "instruction_port" in invalid.errors


def test_llm_local_model_conversion():
    form = LLMServiceForm(llm_data(n_threads=8, cache_type_k="q8_0"), nodes=NODES)
    assert form.is_valid(), form.errors
    assert form.to_spec() == ServiceSpec(
        service_type="llm",
        port=8081,
        settings={
            "bind_host": "0.0.0.0",
            "startup_timeout": 600.0,
            "n_ctx": 32768,
            "n_gpu_layers": -1,
            "n_parallel": 1,
            "model_alias": "local-model",
            "model_path": "/models/model.gguf",
            "extra_args": [
                "--n_threads",
                "8",
                "--n_batch",
                "2048",
                "--n_ubatch",
                "512",
                "--type_k",
                "8",
                "--flash_attn",
                "true",
            ],
        },
    )


def test_llm_hugging_face_conversion_ignores_inactive_local_source():
    form = LLMServiceForm(
        llm_data(
            model_source="huggingface",
            model_path="/models/stale.gguf",
            hf_repo="org/model-GGUF",
            hf_file="model.Q4_K_M.gguf",
            hf_cache_dir="/cache",
        ),
        nodes=NODES,
    )
    assert form.is_valid(), form.errors
    settings = form.to_spec().settings
    assert settings["hf_repo"] == "org/model-GGUF"
    assert settings["hf_file"] == "model.Q4_K_M.gguf"
    assert settings["hf_cache_dir"] == "/cache"
    assert "model_path" not in settings


def test_llm_source_and_batch_validation():
    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"model_path": ""}, "model_path"),
        ({"model_source": "huggingface", "model_path": "", "hf_repo": "", "hf_file": "model.gguf"}, "hf_repo"),
        ({"model_source": "huggingface", "model_path": "", "hf_repo": "org/model", "hf_file": ""}, "hf_file"),
        ({"n_batch": 511, "n_ubatch": 512}, "n_batch"),
    )
    for values, field in invalid_cases:
        form = LLMServiceForm(llm_data(**values), nodes=NODES)
        assert not form.is_valid()
        assert form.errors is not None
        assert field in form.errors

    split_model = LLMServiceForm(
        llm_data(model_path="/models/model-00001-of-00002.gguf"),
        nodes=NODES,
    )
    assert not split_model.is_valid()
    assert "model_path" in split_model.errors


def test_llm_optional_fields_are_omitted():
    form = LLMServiceForm(llm_data(n_threads="", chat_format="", cache_type_k="", cache_type_v=""), nodes=NODES)
    assert form.is_valid(), form.errors
    settings = form.to_spec().settings
    assert "chat_format" not in settings
    assert "n_threads" not in settings
    assert "--n_threads" not in settings["extra_args"]
    assert "--type_k" not in settings["extra_args"]


def test_llm_known_and_quoted_additional_arguments():
    form = LLMServiceForm(llm_data(additional_arguments='--some_new_flag "value with spaces"'), nodes=NODES)
    assert form.is_valid(), form.errors
    assert form.to_spec().settings["extra_args"][:2] == ["--some_new_flag", "value with spaces"]


def test_llm_rejects_protected_additional_arguments():
    form = LLMServiceForm(llm_data(additional_arguments="--model other.gguf"), nodes=NODES)
    assert not form.is_valid()
    assert "additional_arguments" in form.errors


def test_llm_existing_config_populates_known_values_and_preserves_unknown_arguments():
    configured = ConfiguredService(
        "desktop",
        ServiceSpec(
            "llm",
            8082,
            {
                "model_path": "/models/model.gguf",
                "bind_host": "127.0.0.1",
                "startup_timeout": 30,
                "extra_args": ["--n_threads", "8", "--n_batch", "4096", "--some_new_flag", "value"],
            },
        ),
    )
    form = LLMServiceForm(nodes=NODES)
    form.initial_from(configured)
    assert form.initial["model_source"] == "local"
    assert form.initial["n_threads"] == 8
    assert form.initial["n_batch"] == 4096
    assert form.initial["additional_arguments"] == "--some_new_flag value"

    round_trip = LLMServiceForm(
        llm_data(n_threads=8, n_batch=4096, additional_arguments=form.initial["additional_arguments"]),
        nodes=NODES,
    )
    assert round_trip.is_valid(), round_trip.errors
    args = round_trip.to_spec().settings["extra_args"]
    assert args.count("--n_threads") == 1
    assert args.count("--n_batch") == 1
    assert args[:2] == ["--some_new_flag", "value"]


def test_sam_conversion_and_validation():
    form = SAMServiceForm(sam_data(additional_arguments='--device "mps backend"'), nodes=NODES)
    assert form.is_valid(), form.errors
    assert form.to_spec() == ServiceSpec(
        "sam3",
        8090,
        {
            "checkpoint": "/models/sam3.pt",
            "bind_host": "0.0.0.0",
            "startup_timeout": 600.0,
            "confidence": 0.25,
            "extra_args": ["--device", "mps backend"],
        },
    )
    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"checkpoint": ""}, "checkpoint"),
        ({"confidence": 1.1}, "confidence"),
        ({"additional_arguments": "--port 9999"}, "additional_arguments"),
    )
    for values, field in invalid_cases:
        invalid = SAMServiceForm(sam_data(**values), nodes=NODES)
        assert not invalid.is_valid()
        assert invalid.errors is not None
        assert field in invalid.errors


def test_sam_existing_config_populates_initial_values():
    configured = ConfiguredService(
        "desktop",
        ServiceSpec(
            "sam3",
            8091,
            {"checkpoint": "/remote/sam3.pt", "confidence": 0.5, "extra_args": ["--device", "cuda"]},
        ),
    )
    form = SAMServiceForm(nodes=NODES)
    form.initial_from(configured)
    assert form.initial["node"] == "desktop"
    assert form.initial["checkpoint"] == "/remote/sam3.pt"
    assert form.initial["confidence"] == 0.5
    assert form.initial["additional_arguments"] == "--device cuda"
