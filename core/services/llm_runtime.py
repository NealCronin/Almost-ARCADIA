from __future__ import annotations

import fnmatch
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

import requests

from core.errors import ServiceStartupError
from core.services.llm_settings import (
    DRAFT_FLAGS,
    NATIVE_FLAGS,
    PROJECTOR_RE,
    SPLIT_GGUF_RE,
    validate_additional_server_arguments,
)
from core.services.specs import ServiceEndpoint, ServiceSpec
from project.settings import BASE_DIR


class LLMRuntime:
    """Translate one LLM spec into a native llama-server process.

    llama-server generates CLI flags from field names; public server flags
    use the native underscore spellings such as ``--ctx-size`` and
    ``--chat-template``. The server exposes ``/v1/models``.
    """

    _download_locks: dict[str, threading.Lock] = {}
    _download_locks_lock: threading.Lock = threading.Lock()

    @staticmethod
    def models_directory() -> Path:
        return (
            Path(os.environ["ARCADIA_MODELS_DIR"])
            if os.environ.get("ARCADIA_MODELS_DIR")
            else BASE_DIR / "workspace" / "models"
        )

    @classmethod
    def _download_hf_model(cls, repo_id: str, filename: str, cache_subdirectory: str) -> str:
        cache_dir = cls.models_directory() / cache_subdirectory
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / filename
        if target.is_file():
            return str(target)
        lock_key = f"{repo_id}/{filename}"
        with cls._download_locks_lock:
            if lock_key not in cls._download_locks:
                cls._download_locks[lock_key] = threading.Lock()
        lock = cls._download_locks[lock_key]
        with lock:
            if target.is_file():
                return str(target)
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ValueError("Hugging Face model sources require huggingface-hub.") from exc
            return str(hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir, token=None))

    @staticmethod
    def list_repository_files(repo_id: str, *, limit: int = 500) -> list[str]:
        try:
            from huggingface_hub.utils import build_hf_headers
        except ImportError as exc:
            raise ValueError("Hugging Face model sources require huggingface-hub.") from exc
        url = f"https://huggingface.co/api/models/{requests.utils.quote(repo_id, safe='/')}/tree/main"
        try:
            response = requests.get(
                url,
                params={"recursive": "true", "expand": "false", "limit": limit},
                headers=build_hf_headers(token=None),
                timeout=(5, 15),
            )
            response.raise_for_status()
            payload = response.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            messages = {
                401: "private or requires authentication",
                403: "private or requires authentication",
                404: "not found",
                429: "rate limited",
            }
            raise ValueError(f"Hugging Face repository {messages.get(status, 'is unavailable')}.") from exc
        except (requests.RequestException, ValueError) as exc:
            raise ValueError(f"Hugging Face repository is unavailable: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError("Hugging Face repository returned invalid metadata.")
        files = [str(item["path"]) for item in payload if isinstance(item, dict) and isinstance(item.get("path"), str)]
        if len(files) >= limit:
            raise ValueError("Hugging Face repository has too many files to inspect.")
        return files

    @classmethod
    def _select_file(cls, files: list[str], pattern: str | None, *, projector: bool = False) -> str:
        candidates = [
            filename
            for filename in files
            if filename.lower().endswith(".gguf") and bool(PROJECTOR_RE.search(Path(filename).name)) == projector
        ]
        if pattern:
            candidates = [filename for filename in candidates if fnmatch.fnmatchcase(Path(filename).name, pattern)]
        if not candidates:
            kind = "projector" if projector else "model"
            raise ValueError(f"No usable {kind} GGUF matched the selected repository and pattern.")
        # Check if all candidates are split shards
        split_candidates = [f for f in candidates if SPLIT_GGUF_RE.search(Path(f).name)]
        if split_candidates and len(split_candidates) == len(candidates):
            if pattern:
                return candidates[0]  # caller resolves via _resolve_split_or_single
            raise ValueError("Only split GGUF files found. Provide a model file pattern to select one.")
        # Filter out split shards from auto-selection
        candidates = [f for f in candidates if not SPLIT_GGUF_RE.search(Path(f).name)]
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            kind = "projector" if projector else "model"
            raise ValueError(f"No usable {kind} GGUF matched the selected repository and pattern.")
        choices = ", ".join(Path(filename).name for filename in candidates[:20])
        kind = "MMProj file pattern" if projector else "Advanced settings → Model file pattern"
        raise ValueError(f"Multiple usable GGUF files found ({choices}); select one with {kind}.")

    @classmethod
    def _resolve_model_path(cls, settings: dict[str, object]) -> str:
        if settings.get("model_path"):
            raise ValueError(
                "Local model_path is retired; choose a Hugging Face repository and save the model settings."
            )
        repo = settings.get("hf_repo")
        if not isinstance(repo, str) or not repo.strip():
            raise ValueError("LLM service requires a Hugging Face repository.")
        pattern = settings.get("model_file_pattern") or settings.get("hf_file")
        if pattern is not None and not isinstance(pattern, str):
            raise ValueError("Model file pattern must be a string.")
        filename = cls._select_file(cls.list_repository_files(repo), pattern, projector=False)
        shards = cls._resolve_split_or_single(filename, repo, "huggingface")
        return shards[0]

    @classmethod
    def _resolve_projector_path(cls, settings: dict[str, object]) -> str | None:
        if not settings.get("vision_enabled"):
            return None
        repo = settings.get("mmproj_repo") or settings.get("hf_repo")
        if not isinstance(repo, str) or not repo:
            raise ValueError("Vision requires a Hugging Face repository.")
        pattern = settings.get("mmproj_file_pattern")
        if pattern is not None and not isinstance(pattern, str):
            raise ValueError("MMProj file pattern must be a string.")
        filename = cls._select_file(cls.list_repository_files(repo), pattern, projector=True)
        return cls._download_hf_model(repo, filename, "mmproj")

    @classmethod
    def _resolve_draft_path(cls, settings: dict[str, object]) -> str | None:
        if not settings.get("draft_enabled"):
            return None
        repo = settings.get("draft_repo") or settings.get("hf_repo")
        if not isinstance(repo, str) or not repo:
            raise ValueError("Draft model requires a repository.")
        pattern = settings.get("draft_file_pattern")
        if pattern is not None and not isinstance(pattern, str):
            raise ValueError("Draft file pattern must be a string.")
        files = cls.list_repository_files(repo)
        filename = cls._select_file(files, pattern, projector=False)
        resolved = cls._download_hf_model(repo, filename, "huggingface")
        return str(Path(resolved))

    @classmethod
    def _resolve_split_or_single(cls, filepath: str, repo: str, cache_subdir: str) -> list[str]:
        filename = Path(filepath).name
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", filename, re.IGNORECASE)
        if not m:
            return [cls._download_hf_model(repo, filename, cache_subdir)]
        shard_num = int(m.group(1))
        total_shards = int(m.group(2))
        if shard_num != 1:
            raise ValueError(f"Model selection must start from shard 1, got shard {shard_num}")
        prefix = filename[: m.start()]
        all_files = cls.list_repository_files(repo)
        downloaded = []
        for i in range(1, total_shards + 1):
            shard_name = f"{prefix}-{i:05d}-of-{total_shards:05d}.gguf"
            if shard_name not in all_files:
                raise ValueError(f"Split model missing shard {i}/{total_shards}: {shard_name}")
            downloaded.append(cls._download_hf_model(repo, shard_name, cache_subdir))
        return downloaded

    @classmethod
    def _find_executable(cls) -> str:
        env_exe = os.environ.get("ARCADIA_LLAMA_SERVER")
        if env_exe:
            return env_exe
        base_dir = BASE_DIR
        candidates = []
        if sys.platform == "win32":
            candidates.extend(
                [
                    str(base_dir / "vendor" / "llama.cpp" / "build" / "bin" / "Release" / "llama-server.exe"),
                    str(base_dir / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server.exe"),
                ]
            )
        else:
            candidates.append(str(base_dir / "vendor" / "llama.cpp" / "build" / "bin" / "llama-server"))
        import shutil

        which_candidate = shutil.which("llama-server") or (
            shutil.which("llama-server.exe") if sys.platform == "win32" else None
        )
        if which_candidate:
            candidates.append(which_candidate)
        for candidate in candidates:
            if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        raise ServiceStartupError(
            f"No llama-server binary found. Searched: {', '.join(candidates)}. "
            "Set ARCADIA_LLAMA_SERVER or run scripts/install_macos_metal.sh / install_windows_cuda.ps1."
        )

    @classmethod
    def build_command(cls, spec: ServiceSpec, *, allow_test_command: bool = False) -> list[str]:
        settings = spec.settings
        raw_command = settings.get("command")
        if raw_command is not None:
            if (
                not allow_test_command
                or not isinstance(raw_command, list)
                or not all(isinstance(item, str) for item in raw_command)
            ):
                raise ValueError("command is available only to unit tests.")
            return list(raw_command)
        executable = cls._find_executable()
        host = str(settings.get("bind_host", "127.0.0.1"))
        command = [executable, "--host", host, "--port", str(spec.port)]
        command.extend(["--model", cls._resolve_model_path(settings)])
        projector = cls._resolve_projector_path(settings)
        if projector:
            command.extend(["--mmproj", projector])
        for key, flag in NATIVE_FLAGS.items():
            value = settings.get(key)
            if value in (None, ""):
                continue
            # --mmap / --no-mmap: switch-only, no value
            if flag == "--mmap":
                if value:
                    command.append("--mmap")
                else:
                    command.append("--no-mmap")
                continue
            # --mlock: emit only when True, no value
            if flag == "--mlock":
                if value:
                    command.append("--mlock")
                continue
            # --flash-attn: tri-state: auto->omit, on->--flash-attn 1, off->omit
            if flag == "--flash-attn":
                if value in ("auto", ""):
                    continue
                if value in ("on", "1", True):
                    command.extend([flag, "1"])
                continue
            command.extend([flag, str(value)])
        draft_path = cls._resolve_draft_path(settings)
        if draft_path:
            command.extend([DRAFT_FLAGS["draft_model"], draft_path])
            command.extend([DRAFT_FLAGS["draft_method"], str(settings.get("draft_method", "draft-simple"))])
            command.extend([DRAFT_FLAGS["draft_max_tokens"], str(settings.get("draft_max_tokens", 3))])
            command.extend([DRAFT_FLAGS["draft_min_prob"], str(settings.get("draft_min_prob", 0.75))])
            dk = settings.get("draft_cache_type_k", "f16")
            dv = settings.get("draft_cache_type_v", "f16")
            if dk:
                command.extend([DRAFT_FLAGS["draft_cache_type_k"], str(dk)])
            if dv:
                command.extend([DRAFT_FLAGS["draft_cache_type_v"], str(dv)])
        command.extend(validate_additional_server_arguments(settings.get("extra_args", [])))
        return command

    @staticmethod
    def endpoint(spec: ServiceSpec, public_host: str) -> ServiceEndpoint:
        return ServiceEndpoint(
            host=str(spec.settings.get("bind_host", public_host)), port=spec.port, service_type=spec.service_type
        )

    @staticmethod
    def readiness_url(endpoint: ServiceEndpoint) -> str:
        return f"{endpoint.base_url}/health"

    @staticmethod
    def probe(endpoint: ServiceEndpoint, timeout: float) -> requests.Response:
        return requests.get(LLMRuntime.readiness_url(endpoint), timeout=timeout)

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
                raise ServiceStartupError("LLM startup cancelled.")
            if process.poll() is not None:
                raise ServiceStartupError(f"LLM process exited during startup with code {process.returncode}.")
            try:
                response = cls.probe(endpoint, timeout=min(2.0, poll_interval + 0.5))
                if response.status_code == 200:
                    return
                last_error = f"readiness returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            remaining = deadline - time.monotonic()
            if cancel_event is not None:
                if cancel_event.wait(min(poll_interval, max(0.0, remaining))):
                    raise ServiceStartupError("LLM startup cancelled.")
            else:
                time.sleep(poll_interval)
        raise ServiceStartupError(f"LLM readiness timed out: {last_error}")

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
                env=os.environ.copy(),
            )
        except Exception:
            log_handle.close()
            raise
        return process, log_handle, cls.endpoint(spec, public_host)
