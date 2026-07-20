from __future__ import annotations

import math
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

from core.networking import validate_ipv4

DEFAULT_CONTEXT_SIZE = 32768
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1024

SPLIT_GGUF_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
PROJECTOR_RE = re.compile(r"mmproj|^projector", re.IGNORECASE)
REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9_.-]*[A-Za-z0-9])?$"
)


@dataclass(frozen=True, slots=True)
class HFSource:
    repo_id: str
    revision: str = "main"
    filename: str | None = None

    @property
    def exact_file(self) -> bool:
        return self.filename is not None


def validate_hf_repository(value: str) -> str:
    repository = value.strip().strip("/")
    if not REPOSITORY_RE.fullmatch(repository) or ".." in repository:
        raise ValueError(
            "Enter a Hugging Face repository as owner/repository, or paste an exact huggingface.co GGUF link."
        )
    return repository


def _validate_revision(value: str) -> str:
    revision = unquote(value).strip().strip("/") or "main"
    if any(character in revision for character in ("\x00", "\r", "\n")) or revision in (".", ".."):
        raise ValueError("Hugging Face revision is invalid.")
    return revision


def _validate_repo_filename(value: str) -> str:
    filename = unquote(value).strip().lstrip("/")
    path = PurePosixPath(filename)
    if (
        not filename
        or path.is_absolute()
        or ".." in path.parts
        or any(character in filename for character in ("\x00", "\r", "\n"))
        or not filename.lower().endswith(".gguf")
    ):
        raise ValueError("The Hugging Face file link must point to a .gguf file.")
    return filename


def parse_hf_source(value: str) -> HFSource:
    raw = value.strip()
    if not raw:
        raise ValueError("A Hugging Face repository or exact GGUF link is required.")

    if "://" not in raw:
        return HFSource(validate_hf_repository(raw))

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or parsed.netloc.lower() not in {
        "huggingface.co",
        "www.huggingface.co",
    }:
        raise ValueError("Only huggingface.co repository or file links are accepted.")

    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        raise ValueError("Hugging Face link must include owner/repository.")
    repo_id = validate_hf_repository(f"{parts[0]}/{parts[1]}")
    if len(parts) == 2:
        return HFSource(repo_id)

    marker = parts[2]
    if marker in ("blob", "resolve"):
        if len(parts) < 5:
            raise ValueError("Exact Hugging Face file links must include a revision and GGUF filename.")
        revision = _validate_revision(parts[3])
        filename = _validate_repo_filename("/".join(parts[4:]))
        return HFSource(repo_id, revision, filename)
    if marker == "tree":
        revision = _validate_revision(parts[3] if len(parts) >= 4 else "main")
        return HFSource(repo_id, revision)

    raise ValueError("Use an owner/repository value or a huggingface.co blob/resolve link to a GGUF file.")


def format_hf_source(repo_id: str, revision: str = "main", filename: str | None = None) -> str:
    if filename:
        return f"https://huggingface.co/{repo_id}/blob/{revision}/{filename}"
    return repo_id


def parse_additional_server_arguments(value: str) -> list[str]:
    """Parse spaces, commas, and line breaks without invoking a shell.

    Commas inside quoted values are preserved. For example:
    ``--tensor-split "1,1"`` and ``--chat-template-kwargs '{"a":1,"b":2}'``.
    """
    if not value.strip():
        return []
    try:
        lexer = shlex.shlex(value, posix=True, punctuation_chars=",")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = [token for token in lexer if token != ","]
    except ValueError as exc:
        raise ValueError(f"Invalid argument syntax: {exc}") from exc
    return validate_additional_server_arguments(tokens)


# Almost ARCADIA owns these values and always emits them before extra arguments.
OWNED_FLAGS = {
    "-m",
    "--model",
    "-mu",
    "--model-url",
    "-dr",
    "--docker-repo",
    "-hf",
    "-hfr",
    "--hf-repo",
    "-hff",
    "--hf-file",
    "-mm",
    "--mmproj",
    "-mmu",
    "--mmproj-url",
    "--mmproj-auto",
    "--no-mmproj",
    "--no-mmproj-auto",
    "--host",
    "--port",
    "-c",
    "--ctx-size",
    "-n",
    "--predict",
    "--n-predict",
    "--temp",
    "--temperature",
}

# These options alter the control/security protocol or cause a non-serving process.
UNSAFE_FLAGS = {
    "-h",
    "--help",
    "--usage",
    "--version",
    "--completion-bash",
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
    "--embedding",
    "--embeddings",
    "--rerank",
    "--reranking",
}


def _matches_flag(argument: str, blocked: set[str]) -> bool:
    normalized = argument.lower()
    return any(normalized == flag or normalized.startswith(f"{flag}=") for flag in blocked)


def validate_additional_server_arguments(args: object) -> list[str]:
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("Additional llama-server arguments must be a list of strings.")
    validated: list[str] = []
    blocked = OWNED_FLAGS | UNSAFE_FLAGS
    for argument in args:
        if not argument or "\x00" in argument or "\r" in argument or "\n" in argument:
            raise ValueError(
                "Additional arguments cannot contain empty values, null bytes, or line breaks inside one token."
            )
        if _matches_flag(argument, blocked):
            raise ValueError(f"Additional arguments cannot override or enable {argument!r}.")
        validated.append(argument)
    return validated


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def generation_settings(settings: Mapping[str, Any]) -> dict[str, float | int]:
    temperature = _finite_number(settings.get("temperature", DEFAULT_TEMPERATURE), "Temperature")
    max_tokens = _positive_integer(settings.get("max_tokens", DEFAULT_MAX_TOKENS), "Max output tokens")
    if temperature < 0:
        raise ValueError("Temperature must be at least 0.")
    return {"temperature": temperature, "max_tokens": max_tokens}


def validate_llm_settings(settings: Mapping[str, Any], *, remote: bool = False) -> dict[str, Any]:
    if not isinstance(settings, Mapping):
        raise ValueError("LLM settings must be an object.")
    values = dict(settings)
    allowed = {
        "hf_repo",
        "hf_revision",
        "hf_file",
        "mmproj_repo",
        "mmproj_revision",
        "mmproj_file",
        "bind_host",
        "n_ctx",
        "vision_enabled",
        "temperature",
        "max_tokens",
        "model_alias",
        "extra_args",
        "models_cache_subdir",
        # Test-only command is rejected by instruction_server and accepted only by a test controller.
        "command",
    }
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown LLM settings: {', '.join(sorted(unknown))}.")
    if remote and "command" in values:
        raise ValueError("Remote LLM settings cannot include command.")

    values["hf_repo"] = validate_hf_repository(str(values.get("hf_repo", "")))
    values["hf_revision"] = _validate_revision(str(values.get("hf_revision", "main")))
    if values.get("hf_file"):
        values["hf_file"] = _validate_repo_filename(str(values["hf_file"]))
    else:
        values.pop("hf_file", None)

    values["vision_enabled"] = bool(values.get("vision_enabled", False))
    if values["vision_enabled"]:
        values["mmproj_repo"] = validate_hf_repository(str(values.get("mmproj_repo", "")))
        values["mmproj_revision"] = _validate_revision(str(values.get("mmproj_revision", "main")))
        if values.get("mmproj_file"):
            values["mmproj_file"] = _validate_repo_filename(str(values["mmproj_file"]))
        else:
            values.pop("mmproj_file", None)
    else:
        for key in ("mmproj_repo", "mmproj_revision", "mmproj_file"):
            values.pop(key, None)

    values["bind_host"] = validate_ipv4(str(values.get("bind_host", "127.0.0.1")), label="Inference IP")
    values["n_ctx"] = _positive_integer(values.get("n_ctx", DEFAULT_CONTEXT_SIZE), "Context size")
    # Older builds stored an application-forced alias. Ignore it so llama-server
    # keeps its native/model-provided alias unless the user explicitly supplies
    # --alias in Additional arguments.
    values.pop("model_alias", None)
    values["extra_args"] = validate_additional_server_arguments(values.get("extra_args", []))
    # Legacy cache naming was persisted in older configurations. Cache
    # placement is now entirely application-owned.
    values.pop("models_cache_subdir", None)
    values.update(generation_settings(values))
    return values
