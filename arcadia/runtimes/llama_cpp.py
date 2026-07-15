from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

"""LLM runtime that launches llama-cpp-python server."""

import sys
from pathlib import Path
from typing import Callable

from arcadia.contracts import ModelSpec, RunningService, ServiceEndpoint, ServiceSpec
from arcadia.process import ProcessLauncher, ProcessSpec, RunningProcess


# ── Exception ──────────────────────────────────────────────────────────────────

class LLMRuntimeError(RuntimeError):
    """Raised when the LLM runtime cannot start or is misconfigured."""


# ── Runtime ────────────────────────────────────────────────────────────────────

# Keys that are explicitly supported and mapped to CLI flags.
_SUPPORTED_SETTINGS = frozenset({
    "context_size",
    "batch_size",
    "microbatch_size",
    "gpu_layers",
    "threads",
    "threads_batch",
    "flash_attention",
    "host",
    "chat_format",
    "model_projector",
    "extra_args",
})

# Keys that exist in llama.cpp but we do not implement.
_UNSUPPORTED_SETTINGS = frozenset({
    "parallel_slots",
    "image_min_tokens",
})

# Keys that exist in llama.cpp but are not implemented (KV cache quantization).
_KV_CACHE_SETTINGS = frozenset({
    "k_cache_type",
    "v_cache_type",
})


class LLMRuntime:
    """Launches `python -m llama_cpp.server` and returns a `RunningService`."""

    def __init__(
        self,
        process_launcher: ProcessLauncher,
        python_executable: Path | None = None,
        model_downloader: Callable[..., str] | None = None,
        readiness_probe: Callable[[str, int], bool] | None = None,
        startup_timeout: float = 120.0,
        poll_interval: float = 0.25,
    ) -> None:
        self._process_launcher = process_launcher
        self._python_executable = Path(sys.executable) if python_executable is None else Path(python_executable)
        self._model_downloader = model_downloader or self._default_model_downloader
        self._startup_timeout = startup_timeout
        self._poll_interval = poll_interval
        self._readiness_probe = readiness_probe or self._default_readiness_probe

    # ── Model resolution ───────────────────────────────────────────────────────

    @staticmethod
    def _default_model_downloader(repo_id: str, filename: str) -> str:
        """Download a model file from Hugging Face Hub."""
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=repo_id, filename=filename)

    def _resolve_model_path(self, model: ModelSpec) -> Path:
        """Return the local path to the model file."""
        if model.local_path is not None:
            path = Path(model.local_path).expanduser()
            if not path.exists() or not path.is_file():
                raise LLMRuntimeError(f"Model file not found: {path}")
            return path

        if model.repository is None or model.filename is None:
            raise LLMRuntimeError(
                "ModelSpec must provide local_path or repository+filename"
            )

        path_str = self._model_downloader(repo_id=model.repository, filename=model.filename)
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            raise LLMRuntimeError(f"Downloaded model file not found: {path}")
        return path

    # ── Process spec building ──────────────────────────────────────────────────

    def _build_process_spec(
        self, spec: ServiceSpec, model_path: Path
    ) -> ProcessSpec:
        """Build the command-line arguments for `llama_cpp.server`."""
        host = spec.settings.get("host", "127.0.0.1")

        cmd: list[str] = [
            str(self._python_executable),
            "-m",
            "llama_cpp.server",
            "--model",
            str(model_path),
            "--host",
            host,
            "--port",
            str(spec.port),
        ]

        settings = spec.settings

        # Integer settings
        for key, flag in (
            ("context_size", "--n_ctx"),
            ("batch_size", "--n_batch"),
            ("microbatch_size", "--n_ubatch"),
            ("gpu_layers", "--n_gpu_layers"),
            ("threads", "--n_threads"),
            ("threads_batch", "--n_threads_batch"),
        ):
            if key in settings:
                cmd.extend([flag, str(settings[key])])

        # Boolean setting
        if "flash_attention" in settings:
            cmd.extend(["--flash_attn", "true" if settings["flash_attention"] else "false"])

        # String settings
        if "chat_format" in settings:
            cmd.extend(["--chat_format", str(settings["chat_format"])])

        # Model projector
        if "model_projector" in settings:
            proj_path = Path(settings["model_projector"]).expanduser()
            if not proj_path.exists() or not proj_path.is_file():
                raise LLMRuntimeError(f"Model projector not found: {proj_path}")
            cmd.extend(["--clip_model_path", str(proj_path)])

        # Extra args (escape hatch)
        if "extra_args" in settings:
            extra = settings["extra_args"]
            if not isinstance(extra, list):
                raise LLMRuntimeError("extra_args must be a list of strings")
            if not all(isinstance(x, str) for x in extra):
                raise LLMRuntimeError("extra_args must be a list of strings")
            cmd.extend(extra)

        return ProcessSpec(command=cmd)

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate(self, spec: ServiceSpec) -> None:
        """Validate the ServiceSpec before launching."""
        if spec.service_type != "llm":
            raise LLMRuntimeError(
                f"Expected service_type='llm', got '{spec.service_type}'"
            )

        if spec.model is None:
            raise LLMRuntimeError("ModelSpec is required for LLM runtime")

        if not (1 <= spec.port <= 65535):
            raise LLMRuntimeError(f"Invalid port: {spec.port}")

        settings = spec.settings

        # Type checks for supported settings
        type_checks: dict[str, type] = {
            "context_size": int,
            "batch_size": int,
            "microbatch_size": int,
            "gpu_layers": int,
            "threads": int,
            "threads_batch": int,
            "flash_attention": bool,
            "host": str,
            "chat_format": str,
            "model_projector": str,
        }

        for key, expected_type in type_checks.items():
            if key in settings:
                if not isinstance(settings[key], expected_type):
                    raise LLMRuntimeError(
                        f"Setting '{key}' must be {expected_type.__name__}, "
                        f"got {type(settings[key]).__name__}"
                    )

        # String settings must be non-empty
        for key in ("host", "chat_format", "model_projector"):
            if key in settings and not settings[key]:
                raise LLMRuntimeError(f"Setting '{key}' must be non-empty")

        # extra_args must be a list of strings
        if "extra_args" in settings:
            extra = settings["extra_args"]
            if not isinstance(extra, list):
                raise LLMRuntimeError("extra_args must be a list of strings")
            if not all(isinstance(x, str) for x in extra):
                raise LLMRuntimeError("extra_args must be a list of strings")

        # Known unsupported settings
        for key in _UNSUPPORTED_SETTINGS:
            if key in settings:
                raise LLMRuntimeError(f"unsupported setting: {key}")

        # KV cache quantization (not implemented)
        for key in _KV_CACHE_SETTINGS:
            if key in settings:
                raise LLMRuntimeError(
                    "KV cache type settings are not implemented: "
                    "use extra_args for confirmed integer values, or omit"
                )

        # Unknown settings
        unknown = set(settings.keys()) - _SUPPORTED_SETTINGS - _UNSUPPORTED_SETTINGS - _KV_CACHE_SETTINGS
        if unknown:
            raise LLMRuntimeError(f"unknown setting: {sorted(unknown)}")

    # ── Readiness probe ───────────────────────────────────────────────────────

    @staticmethod
    def _default_readiness_probe(host: str, port: int) -> bool:
        """Check if the server is responding on the given host:port."""
        import urllib.request
        import urllib.error

        url = f"http://{host}:{port}/v1/models"
        try:
            with urllib.request.urlopen(url, timeout=2):
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return False

    def _wait_for_service(
        self,
        running_process: RunningProcess,
        host: str,
        port: int,
    ) -> None:
        """Wait until the server is ready or timeout is reached."""
        import time

        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if not self._process_launcher.is_running(running_process):
                stderr_lines = self._process_launcher.recent_stderr(running_process)
                stdout_lines = self._process_launcher.recent_stdout(running_process)
                output = stderr_lines if stderr_lines else stdout_lines
                msg = (
                    f"llama-cpp-python server exited before becoming ready "
                    f"on {host}:{port}.\n\nRecent output:\n"
                    + "\n".join(output)
                )
                raise LLMRuntimeError(msg)
            if self._readiness_probe(host, port):
                return
            time.sleep(self._poll_interval)

        raise LLMRuntimeError(
            f"Server did not become ready within {self._startup_timeout}s "
            f"on {host}:{port}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, spec: ServiceSpec) -> RunningService:
        """Validate, resolve model, launch the server, and return a RunningService."""
        self._validate(spec)

        model_path = self._resolve_model_path(spec.model)

        proc_spec = self._build_process_spec(spec, model_path)

        running_process = self._process_launcher.start(proc_spec)

        host = spec.settings.get("host", "127.0.0.1")

        try:
            self._wait_for_service(running_process, host, spec.port)
        except Exception as exc:
            try:
                self._process_launcher.stop(running_process)
            except Exception:
                logger.exception(
                    "Failed to clean up llama-cpp-python process after startup failure"
                )
            if isinstance(exc, LLMRuntimeError):
                raise
            raise LLMRuntimeError(
                f"Failed while waiting for llama-cpp-python server "
                f"on {host}:{spec.port}"
            ) from exc

        endpoint = ServiceEndpoint(
            host=host,
            port=spec.port,
            service_type=spec.service_type,
        )

        return RunningService(
            spec=spec,
            endpoint=endpoint,
            runtime_handle=running_process,
        )

    def stop(self, service: RunningService) -> None:
        """Stop the LLM server process."""
        if not isinstance(service.runtime_handle, RunningProcess):
            raise LLMRuntimeError(
                "RunningService runtime_handle must be a RunningProcess"
            )
        self._process_launcher.stop(service.runtime_handle)
