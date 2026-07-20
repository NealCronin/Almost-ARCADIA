from __future__ import annotations

import pytest

from core.services.llm_settings import (
    format_hf_source,
    parse_additional_server_arguments,
    parse_hf_source,
    validate_additional_server_arguments,
    validate_llm_settings,
)


def valid_settings(**updates):
    values = {
        "hf_repo": "unsloth/Qwen3.5-2B-GGUF",
        "hf_revision": "main",
        "hf_file": "Qwen3.5-2B-IQ4_XS.gguf",
        "bind_host": "127.0.0.1",
        "n_ctx": 32768,
        "vision_enabled": False,
        "temperature": 0.1,
        "max_tokens": 1024,
        "model_alias": "logical-model",
        "extra_args": ["--flash-attn", "on"],
    }
    values.update(updates)
    return values


def test_parse_repository_source():
    source = parse_hf_source("unsloth/Qwen3.5-2B-GGUF")
    assert source.repo_id == "unsloth/Qwen3.5-2B-GGUF"
    assert source.filename is None


def test_parse_exact_blob_source_with_nested_path():
    source = parse_hf_source("https://huggingface.co/owner/repo/blob/revision-name/quants/model-00001-of-00002.gguf")
    assert source.repo_id == "owner/repo"
    assert source.revision == "revision-name"
    assert source.filename == "quants/model-00001-of-00002.gguf"


def test_format_source_uses_exact_file_link():
    assert format_hf_source("owner/repo", "main", "a.gguf") == "https://huggingface.co/owner/repo/blob/main/a.gguf"


def test_argument_parser_accepts_space_comma_and_lines():
    assert parse_additional_server_arguments("--flash-attn on, --batch-size 2048\n--ubatch-size 512") == [
        "--flash-attn",
        "on",
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
    ]


def test_argument_parser_preserves_quoted_commas_and_json():
    assert parse_additional_server_arguments('--tensor-split "1,1" --chat-template-kwargs \'{"a":1,"b":2}\'') == [
        "--tensor-split",
        "1,1",
        "--chat-template-kwargs",
        '{"a":1,"b":2}',
    ]


@pytest.mark.parametrize("flag", ["--model", "-m", "--host=1.2.3.4", "--ctx-size", "--temp", "--api-key"])
def test_owned_or_unsafe_flags_are_rejected(flag):
    with pytest.raises(ValueError):
        validate_additional_server_arguments([flag])


def test_normal_native_flags_are_passed_through():
    assert validate_additional_server_arguments(["-ngl", "all", "-fa", "on", "--mlock"]) == [
        "-ngl",
        "all",
        "-fa",
        "on",
        "--mlock",
    ]


def test_validate_settings_keeps_only_quick_values_and_drops_old_forced_alias():
    values = validate_llm_settings(valid_settings())
    assert values["temperature"] == 0.1
    assert values["max_tokens"] == 1024
    assert values["extra_args"] == ["--flash-attn", "on"]
    assert "model_alias" not in values


def test_native_alias_is_allowed_in_additional_arguments():
    assert validate_additional_server_arguments(["--alias", "native-name"]) == [
        "--alias",
        "native-name",
    ]


def test_vision_requires_projector_repository():
    with pytest.raises(ValueError):
        validate_llm_settings(valid_settings(vision_enabled=True))
