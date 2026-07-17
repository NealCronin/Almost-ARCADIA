from __future__ import annotations

import ipaddress
import math
import re
import shlex
from collections.abc import Mapping
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from core.networking import local_ipv4_addresses

if TYPE_CHECKING:
    from core.config import NodeConfig


# ============================================================
# Supported values and defaults
# ============================================================

CACHE_TYPE_CHOICES = [
    ("f16", "F16"),
    ("bf16", "BF16"),
    ("q8_0", "Q8_0"),
    ("q5_0", "Q5_0"),
    ("q5_1", "Q5_1"),
    ("q4_0", "Q4_0"),
    ("q4_1", "Q4_1"),
    ("iq4_nl", "IQ4_NL"),
    ("", "Default"),
]

VALID_CACHE_TYPES = {
    value
    for value, _label in CACHE_TYPE_CHOICES
    if value
}

VALID_FLASH_ATTENTION = {
    "auto",
    "on",
    "off",
}

VALID_DRAFT_METHODS = {
    "draft-simple",
    "draft-mtp",
    "draft-eagle3",
}

DRAFT_DEFAULTS: dict[str, Any] = {
    "draft_method": "draft-simple",
    "draft_max_tokens": 3,
    "draft_min_prob": 0.75,
    "draft_cache_type_k": "f16",
    "draft_cache_type_v": "f16",
}

DEFAULT_GENERATION: dict[str, float | int] = {
    "temperature": 0.1,
    "top_k": 20,
    "top_p": 0.9,
    "min_p": 0.05,
    "max_tokens": 1024,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

DEFAULTS: dict[str, Any] = {
    "bind_host": "127.0.0.1",
    "n_ctx": 32768,
    "models_cache_subdir": "huggingface",
    "n_gpu_layers": "all",
    "n_batch": 2048,
    "n_ubatch": 512,
    "flash_attn": "auto",
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "use_mmap": True,
    "use_mlock": False,
    **DEFAULT_GENERATION,
}


# ============================================================
# Repository and file-pattern validation
# ============================================================

REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9]"
    r"(?:[A-Za-z0-9_.-]*[A-Za-z0-9])?"
    r"/"
    r"[A-Za-z0-9]"
    r"(?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$"
)

SPLIT_GGUF_RE = re.compile(
    r"-(\d{5})-of-(\d{5})\.gguf$",
    re.IGNORECASE,
)

PROJECTOR_RE = re.compile(
    r"mmproj|^projector",
    re.IGNORECASE,
)


# ============================================================
# Native llama-server command mapping
# ============================================================

# These are the canonical long-form options emitted by LLMRuntime.
NATIVE_FLAGS = {
    "n_ctx": "--ctx-size",
    "n_gpu_layers": "--n-gpu-layers",
    "n_threads": "--threads",
    "n_batch": "--batch-size",
    "n_ubatch": "--ubatch-size",
    "flash_attn": "--flash-attn",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "use_mmap": "--mmap",
    "use_mlock": "--mlock",
    "model_alias": "--alias",
    "chat_format": "--chat-template",
}

DRAFT_FLAGS = {
    "draft_model": "--spec-draft-model",
    "draft_method": "--spec-type",
    "draft_max_tokens": "--spec-draft-n-max",
    "draft_min_prob": "--spec-draft-p-min",
    "draft_cache_type_k": "--cache-type-k-draft",
    "draft_cache_type_v": "--cache-type-v-draft",
}


# Every native spelling for a setting owned by the typed form/runtime.
# Additional Arguments must not be able to supply any of these.
OWNED_FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "model": (
        "-m",
        "--model",
    ),
    "model_url": (
        "-mu",
        "--model-url",
    ),
    "docker_repo": (
        "-dr",
        "--docker-repo",
    ),
    "hf_repo": (
        "-hf",
        "-hfr",
        "--hf-repo",
    ),
    "hf_file": (
        "-hff",
        "--hf-file",
    ),
    "hf_token": (
        "-hft",
        "--hf-token",
    ),
    "mmproj": (
        "-mm",
        "--mmproj",
        "-mmu",
        "--mmproj-url",
        "--mmproj-auto",
        "--no-mmproj",
        "--no-mmproj-auto",
    ),
    "host": (
        "--host",
        "--bind-host",
    ),
    "port": (
        "--port",
    ),
    "context": (
        "-c",
        "--ctx-size",
    ),
    "gpu_layers": (
        "-ngl",
        "--gpu-layers",
        "--n-gpu-layers",
    ),
    "threads": (
        "-t",
        "--threads",
    ),
    "batch_size": (
        "-b",
        "--batch-size",
    ),
    "microbatch_size": (
        "-ub",
        "--ubatch-size",
    ),
    "flash_attention": (
        "-fa",
        "--flash-attn",
    ),
    "cache_type_k": (
        "-ctk",
        "--cache-type-k",
    ),
    "cache_type_v": (
        "-ctv",
        "--cache-type-v",
    ),
    "mmap": (
        "--mmap",
        "--no-mmap",
    ),
    "mlock": (
        "--mlock",
    ),
    "alias": (
        "-a",
        "--alias",
    ),
    "chat_template": (
        "--chat-template",
        "--chat-template-file",
    ),
    "temperature": (
        "--temp",
        "--temperature",
    ),
    "top_k": (
        "--top-k",
    ),
    "top_p": (
        "--top-p",
    ),
    "min_p": (
        "--min-p",
    ),
    "max_tokens": (
        "-n",
        "--predict",
        "--n-predict",
    ),
    "repeat_penalty": (
        "--repeat-penalty",
    ),
    "presence_penalty": (
        "--presence-penalty",
    ),
    "frequency_penalty": (
        "--frequency-penalty",
    ),
    "seed": (
        "-s",
        "--seed",
    ),
    "draft_repo": (
        "--spec-draft-hf",
        "-hfd",
        "-hfrd",
        "--hf-repo-draft",
    ),
    "draft_model": (
        "--spec-draft-model",
        "-md",
        "--model-draft",
    ),
    "draft_method": (
        "--spec-type",
    ),
    "draft_max_tokens": (
        "--spec-draft-n-max",
    ),
    "draft_min_probability": (
        "--spec-draft-p-min",
        "--draft-p-min",
    ),
    "draft_cache_type_k": (
        "--spec-draft-type-k",
        "-ctkd",
        "--cache-type-k-draft",
    ),
    "draft_cache_type_v": (
        "--spec-draft-type-v",
        "-ctvd",
        "--cache-type-v-draft",
    ),
    # This draft intentionally leaves draft GPU placement automatic.
    "draft_gpu_layers": (
        "--spec-draft-ngl",
        "-ngld",
        "--gpu-layers-draft",
        "--n-gpu-layers-draft",
    ),
    # Removed legacy speculative options should not be reintroduced.
    "removed_draft_options": (
        "--draft",
        "--draft-n",
        "--draft-max",
        "--draft-min",
        "--draft-n-min",
    ),
}

OWNED_FLAGS = {
    flag.lower()
    for aliases in OWNED_FLAG_ALIASES.values()
    for flag in aliases
}


# These are not typed settings, but allowing them would violate the
# application's process, routing, or security boundaries.
UNSAFE_SERVER_FLAGS = {
    "--config",
    "--config-file",
    "--config_file",
    "--api-key",
    "--api-key-file",
    "--ssl-key-file",
    "--ssl-cert-file",
    "--reuse-port",
    "--api-prefix",
    "--tools",
    "-ag",
    "--agent",
    "-no-ag",
    "--no-agent",
    "--ui-mcp-proxy",
    "--webui-mcp-proxy",
    "--no-ui-mcp-proxy",
    "--no-webui-mcp-proxy",
    "--embedding",
    "--embeddings",
    "--rerank",
    "--reranking",
}

DISALLOWED_EXECUTABLES = {
    "bash",
    "bash.exe",
    "cmd",
    "cmd.exe",
    "command",
    "command.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "python",
    "python.exe",
    "python3",
    "python3.exe",
    "sh",
    "sh.exe",
    "zsh",
    "zsh.exe",
}

DISALLOWED_ARGUMENTS = {
    "--",
    "-h",
    "--help",
    "--usage",
    "--command",
    "--python-executable",
    "--python_executable",
    "--server-module",
    "--server_module",
    *DISALLOWED_EXECUTABLES,
}


# ============================================================
# Persisted setting groups
# ============================================================

MODEL_KEYS = {
    "hf_repo",
    "model_file_pattern",
    "models_cache_subdir",
}

VISION_KEYS = {
    "vision_enabled",
    "mmproj_repo",
    "mmproj_file_pattern",
    "chat_format",
}

GENERATION_KEYS = set(DEFAULT_GENERATION) | {
    "seed",
}

DRAFT_SETTING_KEYS = {
    "draft_enabled",
    "draft_repo",
    "draft_file_pattern",
    "draft_method",
    "draft_max_tokens",
    "draft_min_prob",
    "draft_cache_type_k",
    "draft_cache_type_v",
}

# Keys removed when an old configuration is saved through the new form.
RETIRED_SOURCE_KEYS = {
    "model_source",
    "model_path",
    "hf_file",
    "hf_cache_dir",
    "n_parallel",
    "draft_model",
    "draft_model_path",
    "mmproj_path",
    "clip_model_path",
    "startup_timeout",
}

LOCAL_PATH_SOURCE_KEYS = {
    "model_path",
    "draft_model",
    "draft_model_path",
    "mmproj_path",
    "clip_model_path",
}

REMOTE_LLM_KEYS = (
    MODEL_KEYS
    | VISION_KEYS
    | set(NATIVE_FLAGS)
    | DRAFT_SETTING_KEYS
    | GENERATION_KEYS
    | {
        "bind_host",
        "extra_args",
    }
)


# ============================================================
# Bind-host validation
# ============================================================

def resolve_inference_bind_host(
    node_name: str,
    nodes: Mapping[str, NodeConfig],
    submitted_local_bind_host: str | None,
) -> str:
    node = nodes.get(node_name)

    if node is None:
        raise ValueError(
            f"Unknown compute node {node_name!r}."
        )

    if node.mode == "remote":
        try:
            address = ipaddress.IPv4Address(node.host)
        except ipaddress.AddressValueError as exc:
            raise ValueError(
                "Remote inference bind host must be an IPv4 address."
            ) from exc

        if address.is_unspecified:
            raise ValueError(
                "Remote inference bind host cannot be 0.0.0.0."
            )

        return str(address)

    host = (
        submitted_local_bind_host
        or DEFAULTS["bind_host"]
    ).strip()

    try:
        address = ipaddress.IPv4Address(host)
    except ipaddress.AddressValueError as exc:
        raise ValueError(
            "Local inference bind host must be an IPv4 address."
        ) from exc

    if address.is_unspecified:
        raise ValueError(
            "Local inference bind host cannot be 0.0.0.0."
        )

    normalized = str(address)

    if normalized not in local_ipv4_addresses():
        raise ValueError(
            "Local inference bind host must be assigned "
            "to this computer."
        )

    return normalized


# ============================================================
# Hugging Face source validation
# ============================================================

def validate_hf_repository(value: str) -> str:
    repository = value.strip()

    if (
        not REPOSITORY_RE.fullmatch(repository)
        or ".." in repository
        or "--" in repository
        or repository.startswith("-")
    ):
        raise ValueError(
            "Enter a Hugging Face repository as owner/repository."
        )

    return repository


def validate_gguf_pattern(
    value: str | None,
) -> str | None:
    if value is None or not value.strip():
        return None

    pattern = value.strip()

    if (
        "/" in pattern
        or "\\" in pattern
        or not pattern.lower().endswith(".gguf")
        or pattern.startswith("-")
        or ".." in pattern
        or any(
            character in pattern
            for character in (
                ";",
                "|",
                "&",
                "`",
                "$",
                "<",
                ">",
                "\x00",
                "\n",
                "\r",
            )
        )
    ):
        raise ValueError(
            "Model file pattern must be a "
            "basename-only .gguf glob."
        )

    return pattern


# ============================================================
# Additional llama-server arguments
# ============================================================

def _argument_basename(argument: str) -> str:
    normalized = argument.replace("\\", "/")
    return PurePath(normalized).name.lower()


def _matches_blocked_flag(
    argument: str,
    blocked_flags: set[str],
) -> bool:
    normalized = argument.lower()

    return any(
        normalized == flag
        or normalized.startswith(f"{flag}=")
        for flag in blocked_flags
    )


def parse_additional_server_arguments(
    value: str,
) -> list[str]:
    try:
        arguments = shlex.split(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid additional argument syntax: {exc}"
        ) from exc

    return validate_additional_server_arguments(arguments)


def validate_additional_server_arguments(
    args: object,
) -> list[str]:
    if (
        not isinstance(args, list)
        or not all(
            isinstance(argument, str)
            for argument in args
        )
    ):
        raise ValueError(
            "Additional arguments must be a list of strings."
        )

    blocked_flags = OWNED_FLAGS | {
        flag.lower()
        for flag in UNSAFE_SERVER_FLAGS
    }

    validated: list[str] = []

    for argument in args:
        if not argument:
            raise ValueError(
                "Additional arguments cannot contain empty values."
            )

        normalized = argument.lower()
        basename = _argument_basename(argument)

        if (
            normalized in DISALLOWED_ARGUMENTS
            or basename in DISALLOWED_EXECUTABLES
            or _matches_blocked_flag(
                argument,
                blocked_flags,
            )
        ):
            raise ValueError(
                "Additional arguments cannot override or enable "
                f"{argument!r}."
            )

        if any(
            character in argument
            for character in (
                ";",
                "|",
                "&",
                "`",
                "$",
                "<",
                ">",
                "\x00",
                "\n",
                "\r",
            )
        ):
            raise ValueError(
                "Additional arguments cannot contain "
                "shell-related syntax."
            )

        validated.append(argument)

    return validated


# ============================================================
# Numeric helpers
# ============================================================

def _require_integer(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{label} must be an integer."
        )

    if minimum is not None and value < minimum:
        raise ValueError(
            f"{label} must be at least {minimum}."
        )

    if maximum is not None and value > maximum:
        raise ValueError(
            f"{label} must be at most {maximum}."
        )

    return value


def _require_finite_number(
    value: Any,
    *,
    label: str,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(
            f"{label} must be a finite number."
        )

    return float(value)


def _require_boolean(
    value: Any,
    *,
    label: str,
) -> bool:
    if not isinstance(value, bool):
        raise ValueError(
            f"{label} must be true or false."
        )

    return value


def _normalize_cache_type(
    value: Any,
    *,
    label: str,
    default: str = "f16",
) -> str:
    if value in (None, ""):
        return default

    if not isinstance(value, str):
        raise ValueError(
            f"{label} must be a cache-type name."
        )

    normalized = value.strip().lower()

    if normalized not in VALID_CACHE_TYPES:
        choices = ", ".join(
            sorted(VALID_CACHE_TYPES)
        )
        raise ValueError(
            f"{label} must be one of: {choices}."
        )

    return normalized


def _normalize_gpu_layers(
    value: Any,
) -> int | str:
    if value in (None, ""):
        return DEFAULTS["n_gpu_layers"]

    if isinstance(value, bool):
        raise ValueError(
            "GPU layers must be 'auto', 'all', "
            "or a non-negative integer."
        )

    if isinstance(value, int):
        if value == -1:
            return "all"

        if value >= 0:
            return value

        raise ValueError(
            "GPU layers must be 'auto', 'all', "
            "or a non-negative integer."
        )

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {
            "auto",
            "all",
        }:
            return normalized

        if normalized == "-1":
            return "all"

        try:
            integer_value = int(normalized)
        except ValueError as exc:
            raise ValueError(
                "GPU layers must be 'auto', 'all', "
                "or a non-negative integer."
            ) from exc

        if integer_value >= 0:
            return integer_value

    raise ValueError(
        "GPU layers must be 'auto', 'all', "
        "or a non-negative integer."
    )


def _normalize_alias(
    value: Any,
) -> str | None:
    if value in (None, ""):
        return None

    if not isinstance(value, str):
        raise ValueError(
            "Model alias must be a string."
        )

    alias = value.strip()

    if not alias:
        return None

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}",
        alias,
    ):
        raise ValueError(
            "Model alias may contain letters, numbers, "
            "periods, underscores, colons, slashes, and hyphens."
        )

    return alias


def _normalize_optional_text(
    value: Any,
    *,
    label: str,
) -> str | None:
    if value in (None, ""):
        return None

    if not isinstance(value, str):
        raise ValueError(
            f"{label} must be a string."
        )

    normalized = value.strip()

    if not normalized:
        return None

    if any(
        character in normalized
        for character in (
            "\x00",
            "\r",
            "\n",
        )
    ):
        raise ValueError(
            f"{label} cannot contain line breaks."
        )

    return normalized


# ============================================================
# Generation settings
# ============================================================

def generation_settings(
    settings: Mapping[str, Any],
) -> dict[str, float | int]:
    temperature = _require_finite_number(
        settings.get(
            "temperature",
            DEFAULT_GENERATION["temperature"],
        ),
        label="Temperature",
    )
    top_k = _require_integer(
        settings.get(
            "top_k",
            DEFAULT_GENERATION["top_k"],
        ),
        label="Top K",
        minimum=0,
    )
    top_p = _require_finite_number(
        settings.get(
            "top_p",
            DEFAULT_GENERATION["top_p"],
        ),
        label="Top P",
    )
    min_p = _require_finite_number(
        settings.get(
            "min_p",
            DEFAULT_GENERATION["min_p"],
        ),
        label="Min P",
    )
    max_tokens = _require_integer(
        settings.get(
            "max_tokens",
            DEFAULT_GENERATION["max_tokens"],
        ),
        label="Max tokens",
        minimum=1,
    )
    repeat_penalty = _require_finite_number(
        settings.get(
            "repeat_penalty",
            DEFAULT_GENERATION["repeat_penalty"],
        ),
        label="Repeat penalty",
    )
    presence_penalty = _require_finite_number(
        settings.get(
            "presence_penalty",
            DEFAULT_GENERATION["presence_penalty"],
        ),
        label="Presence penalty",
    )
    frequency_penalty = _require_finite_number(
        settings.get(
            "frequency_penalty",
            DEFAULT_GENERATION["frequency_penalty"],
        ),
        label="Frequency penalty",
    )

    if temperature < 0:
        raise ValueError(
            "Temperature must be at least 0."
        )

    if not 0 <= top_p <= 1:
        raise ValueError(
            "Top P must be between 0 and 1."
        )

    if not 0 <= min_p <= 1:
        raise ValueError(
            "Min P must be between 0 and 1."
        )

    if repeat_penalty <= 0:
        raise ValueError(
            "Repeat penalty must be greater than zero."
        )

    values: dict[str, float | int] = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": min_p,
        "max_tokens": max_tokens,
        "repeat_penalty": repeat_penalty,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }

    if (
        "seed" in settings
        and settings["seed"] is not None
    ):
        values["seed"] = _require_integer(
            settings["seed"],
            label="Seed",
        )

    return values


# ============================================================
# Complete LLM setting validation
# ============================================================

def validate_llm_settings(
    settings: Mapping[str, Any],
    *,
    remote: bool = False,
) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise ValueError(
            "LLM settings must be an object."
        )

    values = dict(settings)

    if remote:
        unknown = set(values) - REMOTE_LLM_KEYS

        if unknown:
            raise ValueError(
                "Remote LLM settings cannot include: "
                f"{', '.join(sorted(unknown))}"
            )

    # Local filesystem model/projector/draft sources are retired.
    for key in LOCAL_PATH_SOURCE_KEYS:
        value = values.get(key)

        if value not in (
            None,
            "",
        ):
            raise ValueError(
                f"{key} is retired; configure a Hugging Face "
                "repository instead."
            )

    # Preserve migration compatibility for old hf_repo + hf_file
    # configurations while normalizing them into the new field.
    legacy_hf_file = values.get("hf_file")

    if (
        legacy_hf_file
        and not values.get("model_file_pattern")
    ):
        values["model_file_pattern"] = legacy_hf_file

    for key in RETIRED_SOURCE_KEYS:
        values.pop(key, None)

    values["hf_repo"] = validate_hf_repository(
        str(values.get("hf_repo", ""))
    )

    for key in (
        "model_file_pattern",
        "mmproj_file_pattern",
        "draft_file_pattern",
    ):
        pattern = validate_gguf_pattern(
            values.get(key)
        )

        if pattern is None:
            values.pop(key, None)
        else:
            values[key] = pattern

    for key in (
        "mmproj_repo",
        "draft_repo",
    ):
        repository = values.get(key)

        if repository in (
            None,
            "",
        ):
            values.pop(key, None)
        else:
            values[key] = validate_hf_repository(
                str(repository)
            )

    values["models_cache_subdir"] = "huggingface"

    bind_host_value = str(
        values.get(
            "bind_host",
            DEFAULTS["bind_host"],
        )
    )

    try:
        bind_address = ipaddress.IPv4Address(
            bind_host_value
        )
    except ipaddress.AddressValueError as exc:
        raise ValueError(
            "Inference bind host must be an IPv4 address."
        ) from exc

    if bind_address.is_unspecified:
        raise ValueError(
            "Inference bind host cannot be 0.0.0.0."
        )

    values["bind_host"] = str(bind_address)

    values["vision_enabled"] = _require_boolean(
        values.get(
            "vision_enabled",
            False,
        ),
        label="Enable vision",
    )
    values["draft_enabled"] = _require_boolean(
        values.get(
            "draft_enabled",
            False,
        ),
        label="Enable draft model",
    )
    values["use_mmap"] = _require_boolean(
        values.get(
            "use_mmap",
            DEFAULTS["use_mmap"],
        ),
        label="Use mmap",
    )
    values["use_mlock"] = _require_boolean(
        values.get(
            "use_mlock",
            DEFAULTS["use_mlock"],
        ),
        label="Use mlock",
    )

    values["n_ctx"] = _require_integer(
        values.get(
            "n_ctx",
            DEFAULTS["n_ctx"],
        ),
        label="Context size",
        minimum=1,
    )
    values["n_gpu_layers"] = _normalize_gpu_layers(
        values.get("n_gpu_layers")
    )

    if values.get("n_threads") in (
        None,
        "",
    ):
        values.pop("n_threads", None)
    else:
        values["n_threads"] = _require_integer(
            values["n_threads"],
            label="CPU threads",
            minimum=1,
        )

    values["n_batch"] = _require_integer(
        values.get(
            "n_batch",
            DEFAULTS["n_batch"],
        ),
        label="Batch size",
        minimum=1,
    )
    values["n_ubatch"] = _require_integer(
        values.get(
            "n_ubatch",
            DEFAULTS["n_ubatch"],
        ),
        label="Microbatch size",
        minimum=1,
    )

    if values["n_ubatch"] > values["n_batch"]:
        raise ValueError(
            "Microbatch size cannot exceed batch size."
        )

    flash_attention = values.get(
        "flash_attn",
        DEFAULTS["flash_attn"],
    )

    if not isinstance(flash_attention, str):
        raise ValueError(
            "Flash attention must be auto, on, or off."
        )

    flash_attention = flash_attention.strip().lower()

    if flash_attention not in VALID_FLASH_ATTENTION:
        raise ValueError(
            "Flash attention must be auto, on, or off."
        )

    values["flash_attn"] = flash_attention

    values["cache_type_k"] = _normalize_cache_type(
        values.get("cache_type_k"),
        label="K-cache type",
    )
    values["cache_type_v"] = _normalize_cache_type(
        values.get("cache_type_v"),
        label="V-cache type",
    )

    alias = _normalize_alias(
        values.get("model_alias")
    )

    if alias is None:
        values.pop("model_alias", None)
    else:
        values["model_alias"] = alias

    chat_format = _normalize_optional_text(
        values.get("chat_format"),
        label="Chat format / template",
    )

    if chat_format is None:
        values.pop("chat_format", None)
    else:
        values["chat_format"] = chat_format

    draft_method = values.get(
        "draft_method",
        DRAFT_DEFAULTS["draft_method"],
    )

    if not isinstance(draft_method, str):
        raise ValueError(
            "Draft method must be a string."
        )

    draft_method = draft_method.strip().lower()

    if draft_method not in VALID_DRAFT_METHODS:
        raise ValueError(
            "Draft method must be draft-simple, "
            "draft-mtp, or draft-eagle3."
        )

    values["draft_method"] = draft_method
    values["draft_max_tokens"] = _require_integer(
        values.get(
            "draft_max_tokens",
            DRAFT_DEFAULTS["draft_max_tokens"],
        ),
        label="Draft max tokens",
        minimum=1,
    )

    draft_min_probability = _require_finite_number(
        values.get(
            "draft_min_prob",
            DRAFT_DEFAULTS["draft_min_prob"],
        ),
        label="Draft minimum probability",
    )

    if not 0 <= draft_min_probability <= 1:
        raise ValueError(
            "Draft minimum probability must be "
            "between 0 and 1."
        )

    values["draft_min_prob"] = (
        draft_min_probability
    )
    values["draft_cache_type_k"] = (
        _normalize_cache_type(
            values.get("draft_cache_type_k"),
            label="Draft K-cache type",
        )
    )
    values["draft_cache_type_v"] = (
        _normalize_cache_type(
            values.get("draft_cache_type_v"),
            label="Draft V-cache type",
        )
    )

    extra_arguments = (
        values.get("extra_args", [])
    )
    validated_extra_arguments = (
        validate_additional_server_arguments(
            extra_arguments
        )
    )

    if validated_extra_arguments:
        values["extra_args"] = (
            validated_extra_arguments
        )
    else:
        values.pop("extra_args", None)

    values.update(
        generation_settings(values)
    )

    return values