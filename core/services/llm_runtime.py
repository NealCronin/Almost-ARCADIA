from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO

import requests

from core.errors import ServiceStartupError
from core.services.llm_settings import PROJECTOR_RE, RUNTIME_FLAGS, SPLIT_GGUF_RE, validate_additional_server_arguments
from core.services.specs import ServiceEndpoint, ServiceSpec
from project.settings import BASE_DIR


class LLMRuntime:
    """Translate one LLM spec into the pinned llama-cpp-python server process.

    llama-cpp-python 0.3.34 generates CLI flags from Pydantic field names, so
    its public server flags intentionally use underscore spellings such as
    ``--n_ctx`` and ``--chat_format``. The server exposes ``/v1/models``.
    """

    @staticmethod
    def models_directory() -> Path:
        return (
            Path(os.environ["ARCADIA_MODELS_DIR"])
            if os.environ.get("ARCADIA_MODELS_DIR")
            else BASE_DIR / "workspace" / "models"
        )

    @classmethod
    def _download_hf_model(cls, repo_id: str, filename: str, cache_subdirectory: str) -> str:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ValueError("Hugging Face model sources require huggingface-hub.") from exc
        cache_dir = cls.models_directory() / cache_subdirectory
        cache_dir.mkdir(parents=True, exist_ok=True)
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
            if filename.lower().endswith(".gguf")
            and bool(PROJECTOR_RE.search(Path(filename).name)) == projector
            and not SPLIT_GGUF_RE.search(Path(filename).name)
        ]
        if pattern:
            candidates = [filename for filename in candidates if fnmatch.fnmatchcase(Path(filename).name, pattern)]
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
        return cls._download_hf_model(repo, filename, "huggingface")

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
        host = str(settings.get("bind_host", "127.0.0.1"))
        command = [sys.executable, "-m", "llama_cpp.server", "--host", host, "--port", str(spec.port)]
        command.extend(["--model", cls._resolve_model_path(settings)])
        projector = cls._resolve_projector_path(settings)
        if projector:
            command.extend(["--clip_model_path", projector])
        for key, flag in RUNTIME_FLAGS.items():
            value = settings.get(key)
            if value not in (None, ""):
                command.extend([flag, str(value).lower() if isinstance(value, bool) else str(value)])
        command.extend(validate_additional_server_arguments(settings.get("extra_args", [])))
        return command

    @staticmethod
    def endpoint(spec: ServiceSpec, public_host: str) -> ServiceEndpoint:
        return ServiceEndpoint(
            host=str(spec.settings.get("bind_host", public_host)), port=spec.port, service_type="llm"
        )

    @staticmethod
    def readiness_url(endpoint: ServiceEndpoint) -> str:
        return f"{endpoint.base_url}/v1/models"

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
