from __future__ import annotations

import ipaddress
import math
import re
import shlex
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from core.networking import local_ipv4_addresses

if TYPE_CHECKING:
    from core.config import NodeConfig

CACHE_TYPE_CHOICES = [
    ("f16", "F16"), ("bf16", "BF16"), ("q8_0", "Q8_0"), ("q5_0", "Q5_0"),
    ("q5_1", "Q5_1"), ("q4_0", "Q4_0"), ("q4_1", "Q4_1"), ("iq4_nl", "IQ4_NL"),
    ("", "Default"),
]

DRAFT_DEFAULTS = {
    "draft_method": "draft-simple",
    "draft_max_tokens": 3,
    "draft_min_prob": 0.75,
    "draft_cache_type_k": "f16",
    "draft_cache_type_v": "f16",
}

DEFAULT_GENERATION = {"temperature": 0.2, "top_k": 40, "min_p": 0.05, "top_p": 0.95}
DEFAULTS = {
    "bind_host": "127.0.0.1",
    "n_ctx": 32768,
    "models_cache_subdir": "huggingface",
    **DEFAULT_GENERATION,
}
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$")
SPLIT_GGUF_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$", re.IGNORECASE)
PROJECTOR_RE = re.compile(r"mmproj|^projector", re.IGNORECASE)

NATIVE_FLAGS = {
    "n_ctx": "--ctx-size",
    "n_gpu_layers": "--n-gpu-layers",
    "n_threads": "--threads",
    "n_batch": "--batch-size",
    "n_ubatch": "--ubatch-size",
    "flash_attn": "--flash-attn",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "use_mmap": "--no-mmap",
    "use_mlock": "--mlock",
    "model_alias": "--alias",
    "chat_format": "--chat-template",
}
MODEL_KEYS = {"hf_repo", "model_file_pattern", "models_cache_subdir"}
VISION_KEYS = {"vision_enabled", "mmproj_repo", "mmproj_file_pattern", "chat_format"}
GENERATION_KEYS = set(DEFAULT_GENERATION)
RETIRED_SOURCE_KEYS = {"model_source", "model_path", "hf_file", "hf_cache_dir", "n_parallel"}
OWNED_FLAGS = set(NATIVE_FLAGS.values()) | {
    "--model",
    "--mmproj",
    "--host",
    "--port",
    "--hf-repo",
    "--hf-file",
    "--bind-host",
    "--draft-model",
    "--speculative",
    "--draft-max",
    "--draft-min",
    "--draft-cache-type-k",
    "--draft-cache-type-v",
}
DISALLOWED_ARGUMENTS = {
    "--",
    "-c",
    "-m",
    "-h",
    "--help",
    "--command",
    "--config_file",
    "--python_executable",
    "--server_module",
    "bash",
    "cmd",
    "powershell",
    "python",
    "python3",
    "sh",
    "zsh",
}
REMOTE_LLM_KEYS = (
    MODEL_KEYS | VISION_KEYS | set(NATIVE_FLAGS) | GENERATION_KEYS | {"bind_host", "extra_args"}
)


def resolve_inference_bind_host(
    node_name: str, nodes: Mapping[str, "NodeConfig"], submitted_local_bind_host: str | None
) -> str:
    node = nodes.get(node_name)
    if node is None:
        raise ValueError(f"Unknown compute node {node_name!r}.")
    if node.mode == "remote":
        try:
            return str(ipaddress.IPv4Address(node.host))
        except ipaddress.AddressValueError as exc:
            raise ValueError("Remote inference bind host must be an IPv4 address.") from exc
    host = (submitted_local_bind_host or "127.0.0.1").strip()
    try:
        host = str(ipaddress.IPv4Address(host))
    except ipaddress.AddressValueError as exc:
        raise ValueError("Local inference bind host must be an IPv4 address.") from exc
    if host not in local_ipv4_addresses():
        raise ValueError("Local inference bind host must be assigned to this computer.")
    return host


def validate_hf_repository(value: str) -> str:
    repository = value.strip()
    if (
        not REPOSITORY_RE.fullmatch(repository)
        or ".." in repository
        or "--" in repository
        or repository.startswith("-")
    ):
        raise ValueError("Enter a Hugging Face repository as owner/repository.")
    return repository


def validate_gguf_pattern(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    pattern = value.strip()
    if (
        "/" in pattern
        or "\\" in pattern
        or not pattern.lower().endswith(".gguf")
        or pattern.startswith("-")
        or any(char in pattern for char in ";|&`$<>")
        or ".." in pattern
    ):
        raise ValueError("Model file pattern must be a basename-only .gguf glob.")
    return pattern


def parse_additional_server_arguments(value: str) -> list[str]:
    try:
        return validate_additional_server_arguments(shlex.split(value))
    except ValueError as exc:
        raise ValueError(f"Invalid additional argument syntax: {exc}") from exc


def validate_additional_server_arguments(args: list[str]) -> list[str]:
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("Additional arguments must be a list of strings.")
    for arg in args:
        if (
            not arg
            or arg in DISALLOWED_ARGUMENTS
            or any(arg == flag or arg.startswith(f"{flag}=") for flag in OWNED_FLAGS)
        ):
            raise ValueError(f"Additional arguments cannot override {arg!r}.")
        if any(char in arg for char in ";|&`$<>"):
            raise ValueError("Additional arguments cannot contain shell-related syntax.")
    return list(args)


def generation_settings(settings: Mapping[str, Any]) -> dict[str, float | int]:
    values = dict(DEFAULT_GENERATION)
    for key, default in DEFAULT_GENERATION.items():
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number.")
        values[key] = int(value) if isinstance(default, int) else float(value)
    if (
        values["temperature"] < 0
        or values["top_k"] < 0
        or not 0 <= values["min_p"] <= 1
        or not 0 < values["top_p"] <= 1
    ):
        raise ValueError("Generation settings are out of range.")
    return values


def validate_llm_settings(settings: Mapping[str, Any], *, remote: bool = False) -> dict[str, Any]:
    values = dict(settings)
    if remote:
        unknown = set(values) - REMOTE_LLM_KEYS
        if unknown:
            raise ValueError(f"Remote LLM settings cannot include: {', '.join(sorted(unknown))}")
    values["hf_repo"] = validate_hf_repository(str(values.get("hf_repo", "")))
    for key in ("model_file_pattern", "mmproj_file_pattern"):
        values[key] = validate_gguf_pattern(values.get(key))
        if values[key] is None:
            values.pop(key, None)
    if values.get("mmproj_repo"):
        values["mmproj_repo"] = validate_hf_repository(str(values["mmproj_repo"]))
    values["models_cache_subdir"] = "huggingface"
    values["vision_enabled"] = bool(values.get("vision_enabled", False))
    values["bind_host"] = str(ipaddress.IPv4Address(str(values.get("bind_host", DEFAULTS["bind_host"]))))
    values["extra_args"] = validate_additional_server_arguments(values.get("extra_args", []))
    values.update(generation_settings(values))
    return values
