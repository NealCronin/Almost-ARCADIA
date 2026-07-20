from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path, PurePosixPath
from typing import IO

import requests

from core.errors import ServiceStartupError
from core.services.llm_settings import (
    PROJECTOR_RE,
    SPLIT_GGUF_RE,
    validate_additional_server_arguments,
    validate_llm_settings,
)
from core.services.specs import ServiceEndpoint, ServiceSpec
from project.settings import BASE_DIR


class LLMRuntime:
    """Translate one saved LLM service into one native llama-server process."""

    _download_locks: dict[str, threading.Lock] = {}
    _download_locks_lock = threading.Lock()

    @staticmethod
    def huggingface_directory() -> Path:
        """Return the compute node's dedicated Hugging Face cache root.

        ``ARCADIA_HUGGINGFACE_DIR`` takes precedence. ``ARCADIA_MODELS_DIR`` is
        retained as a deprecated alias for the complete Hugging Face root, not
        just the models child directory.
        """
        override = os.environ.get("ARCADIA_HUGGINGFACE_DIR")
        legacy_override = os.environ.get("ARCADIA_MODELS_DIR")
        if override:
            root = Path(override).expanduser()
        elif legacy_override:
            warnings.warn(
                "ARCADIA_MODELS_DIR is deprecated; use ARCADIA_HUGGINGFACE_DIR instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            root = Path(legacy_override).expanduser()
        else:
            root = BASE_DIR / "huggingface"
            legacy_root = BASE_DIR / "workspace" / "huggingface"
            if legacy_root.exists():
                warnings.warn(
                    f"Existing Hugging Face cache at {legacy_root} was not migrated; use {root} for new files.",
                    UserWarning,
                    stacklevel=2,
                )
        root.mkdir(parents=True, exist_ok=True)
        for child in ("models", "mmproj"):
            (root / child).mkdir(exist_ok=True)
        return root

    @classmethod
    def models_directory(cls) -> Path:
        """Compatibility alias for callers that need the Hugging Face root."""
        return cls.huggingface_directory()

    @classmethod
    def cache_directory(cls, cache_kind: str) -> Path:
        if cache_kind not in {"models", "mmproj"}:
            raise ValueError(f"Unknown Hugging Face cache kind: {cache_kind!r}.")
        path = cls.huggingface_directory() / cache_kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def _download_hf_model(cls, repo_id: str, revision: str, filename: str, cache_subdir: str) -> str:
        root = cls.cache_directory(cache_subdir)
        lock_key = f"{repo_id}@{revision}:{filename}"
        with cls._download_locks_lock:
            lock = cls._download_locks.setdefault(lock_key, threading.Lock())
        with lock:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ValueError("Hugging Face sources require huggingface-hub.") from exc
            try:
                return str(
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        revision=revision,
                        cache_dir=root,
                    )
                )
            except Exception as exc:
                raise ValueError(f"Could not download {repo_id}@{revision}/{filename}: {exc}") from exc

    @staticmethod
    def list_repository_files(repo_id: str, *, revision: str = "main", limit: int = 1000) -> list[str]:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ValueError("Hugging Face sources require huggingface-hub.") from exc
        try:
            files = list(HfApi().list_repo_files(repo_id=repo_id, revision=revision))
        except Exception as exc:
            raise ValueError(f"Hugging Face repository is unavailable: {exc}") from exc
        if len(files) > limit:
            raise ValueError(f"Hugging Face repository contains more than {limit} files; use an exact GGUF file link.")
        return [str(item) for item in files]

    @staticmethod
    def _is_projector(filename: str) -> bool:
        return bool(PROJECTOR_RE.search(PurePosixPath(filename).name))

    @classmethod
    def _select_file(
        cls,
        files: list[str],
        exact_filename: str | None,
        *,
        projector: bool,
    ) -> str:
        candidates = [item for item in files if item.lower().endswith(".gguf") and cls._is_projector(item) == projector]
        kind = "projector" if projector else "model"
        if exact_filename:
            matches = [item for item in candidates if item.casefold() == exact_filename.casefold()]
            if len(matches) != 1:
                raise ValueError(f"The exact {kind} file {exact_filename!r} was not found in the selected repository.")
            selected = matches[0]
            match = SPLIT_GGUF_RE.search(PurePosixPath(selected).name)
            if match and int(match.group(1)) != 1:
                raise ValueError(f"Select the first split shard for the {kind}, not shard {match.group(1)}.")
            return selected

        unsplit = [item for item in candidates if SPLIT_GGUF_RE.search(PurePosixPath(item).name) is None]
        first_shards: list[str] = []
        for item in candidates:
            match = SPLIT_GGUF_RE.search(PurePosixPath(item).name)
            if match and int(match.group(1)) == 1:
                first_shards.append(item)
        selectable = unsplit + first_shards
        if len(selectable) == 1:
            return selectable[0]
        if not selectable:
            raise ValueError(f"No usable {kind} GGUF exists in the selected repository.")
        choices = ", ".join(selectable[:12])
        raise ValueError(
            f"The repository contains multiple usable {kind} GGUF files ({choices}). "
            "Paste the exact huggingface.co blob link for the file you want."
        )

    @classmethod
    def _resolve_split_or_single(
        cls,
        selected: str,
        repo_id: str,
        revision: str,
        cache_subdir: str,
        files: list[str],
    ) -> list[str]:
        path = PurePosixPath(selected)
        match = SPLIT_GGUF_RE.search(path.name)
        if not match:
            return [cls._download_hf_model(repo_id, revision, selected, cache_subdir)]
        shard_number = int(match.group(1))
        total = int(match.group(2))
        if shard_number != 1:
            raise ValueError("Split GGUF selection must point to shard 00001.")
        prefix = path.name[: match.start()]
        parent = "" if str(path.parent) == "." else f"{path.parent.as_posix()}/"
        downloaded: list[str] = []
        file_set = set(files)
        for index in range(1, total + 1):
            name = f"{parent}{prefix}-{index:05d}-of-{total:05d}.gguf"
            if name not in file_set:
                raise ValueError(f"Split GGUF is missing shard {index}/{total}: {name}")
            downloaded.append(cls._download_hf_model(repo_id, revision, name, cache_subdir))
        return downloaded

    @classmethod
    def _resolve_model_path(cls, settings: dict[str, object]) -> str:
        repo = str(settings["hf_repo"])
        revision = str(settings.get("hf_revision", "main"))
        exact = settings.get("hf_file")
        files = cls.list_repository_files(repo, revision=revision)
        selected = cls._select_file(files, str(exact) if exact else None, projector=False)
        return cls._resolve_split_or_single(selected, repo, revision, "models", files)[0]

    @classmethod
    def _resolve_projector_path(cls, settings: dict[str, object]) -> str | None:
        if not settings.get("vision_enabled"):
            return None
        repo = str(settings["mmproj_repo"])
        revision = str(settings.get("mmproj_revision", "main"))
        exact = settings.get("mmproj_file")
        files = cls.list_repository_files(repo, revision=revision)
        selected = cls._select_file(files, str(exact) if exact else None, projector=True)
        return cls._resolve_split_or_single(selected, repo, revision, "mmproj", files)[0]

    @classmethod
    def _find_executable(cls) -> str:
        override = os.environ.get("ARCADIA_LLAMA_SERVER")
        if override:
            path = Path(override).expanduser()
            if not path.is_file():
                raise ServiceStartupError(f"ARCADIA_LLAMA_SERVER does not exist: {path}")
            return str(path)
        candidates: list[Path] = []
        if sys.platform == "win32":
            candidates.extend(
                [
                    BASE_DIR / "vendor" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe",
                    BASE_DIR / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server.exe",
                ]
            )
        else:
            candidates.append(BASE_DIR / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server")
        discovered = shutil.which("llama-server") or (
            shutil.which("llama-server.exe") if sys.platform == "win32" else None
        )
        if discovered:
            candidates.append(Path(discovered))
        for candidate in candidates:
            if candidate.is_file() and (sys.platform == "win32" or os.access(candidate, os.X_OK)):
                return str(candidate)
        raise ServiceStartupError(
            "No native llama-server binary was found. Set ARCADIA_LLAMA_SERVER or run the platform install script."
        )

    @classmethod
    def build_command(cls, spec: ServiceSpec, *, allow_test_command: bool = False) -> list[str]:
        raw_command = spec.settings.get("command")
        if raw_command is not None:
            if (
                not allow_test_command
                or not isinstance(raw_command, list)
                or not all(isinstance(item, str) for item in raw_command)
            ):
                raise ValueError("command is available only to unit tests.")
            return list(raw_command)

        settings = validate_llm_settings(spec.settings)
        command = [
            cls._find_executable(),
            "--host",
            str(settings["bind_host"]),
            "--port",
            str(spec.port),
            "--model",
            cls._resolve_model_path(settings),
            "--ctx-size",
            str(settings["n_ctx"]),
        ]
        projector = cls._resolve_projector_path(settings)
        if projector:
            command.extend(["--mmproj", projector])
        command.extend(validate_additional_server_arguments(settings.get("extra_args", [])))
        return command

    @staticmethod
    def endpoint(spec: ServiceSpec, public_host: str) -> ServiceEndpoint:
        return ServiceEndpoint(
            host=str(spec.settings.get("bind_host", public_host)),
            port=spec.port,
            service_type=spec.service_type,
        )

    @staticmethod
    def readiness_url(endpoint: ServiceEndpoint) -> str:
        return f"{endpoint.base_url}/health"

    @classmethod
    def wait_ready(
        cls,
        process: subprocess.Popen[str],
        endpoint: ServiceEndpoint,
        *,
        timeout: float,
        poll_interval: float = 0.5,
        cancel_event: threading.Event | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = "service is still loading"
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise ServiceStartupError(f"{endpoint.service_type} startup cancelled.")
            return_code = process.poll()
            if return_code is not None:
                raise ServiceStartupError(f"{endpoint.service_type} exited during startup with code {return_code}.")
            try:
                response = requests.get(cls.readiness_url(endpoint), timeout=2)
                if response.ok:
                    return
                last_error = f"health returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(poll_interval)
        raise ServiceStartupError(f"{endpoint.service_type} did not become ready within {timeout:g}s: {last_error}")

    @classmethod
    def launch(
        cls,
        spec: ServiceSpec,
        *,
        public_host: str,
        log_path: Path,
        allow_test_command: bool = False,
    ) -> tuple[subprocess.Popen[str], IO[str], ServiceEndpoint]:
        command = cls.build_command(spec, allow_test_command=allow_test_command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
        except Exception:
            log_handle.close()
            raise
        return process, log_handle, cls.endpoint(spec, public_host)
