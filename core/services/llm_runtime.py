from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO

import requests

from core.errors import ServiceStartupError
from core.services.specs import ServiceEndpoint, ServiceSpec


class LLMRuntime:
    """Translate LLM service settings into an owned llama-cpp server process."""

    @staticmethod
    def build_command(spec: ServiceSpec, *, allow_test_command: bool = False) -> list[str]:
        settings = spec.settings
        raw_command = settings.get("command")
        if raw_command is not None:
            if not allow_test_command:
                raise ValueError("command is available only to unit tests.")
            if not isinstance(raw_command, list) or not all(isinstance(item, str) for item in raw_command):
                raise ValueError("command must be a list of strings.")
            return list(raw_command)

        executable = str(settings.get("python_executable", sys.executable))
        command = [executable, "-m", str(settings.get("server_module", "llama_cpp.server"))]
        command.extend(["--host", str(settings.get("bind_host", "0.0.0.0"))])
        command.extend(["--port", str(spec.port)])

        hf_repo = settings.get("hf_repo")
        hf_file = settings.get("hf_file")
        model_path = settings.get("model_path")
        if model_path:
            command.extend(["--model", str(model_path)])
        elif hf_repo and hf_file:
            command.extend(["--hf_repo", str(hf_repo), "--hf_file", str(hf_file)])
        else:
            raise ValueError("LLM service requires model_path or hf_repo plus hf_file.")

        for key, flag in {
            "n_ctx": "--n_ctx",
            "n_gpu_layers": "--n_gpu_layers",
            "chat_format": "--chat_format",
            "model_alias": "--model_alias",
        }.items():
            if key in settings and settings[key] is not None:
                command.extend([flag, str(settings[key])])
        extra_args = settings.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
            raise ValueError("extra_args must be a list of strings.")
        command.extend(extra_args)
        return command

    @staticmethod
    def endpoint(spec: ServiceSpec, public_host: str) -> ServiceEndpoint:
        return ServiceEndpoint(host=public_host, port=spec.port, service_type="llm")

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
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = "service is still loading"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ServiceStartupError(f"LLM process exited during startup with code {process.returncode}.")
            try:
                response = cls.probe(endpoint, timeout=min(2.0, poll_interval + 0.5))
                if response.status_code == 200:
                    return
                last_error = f"readiness returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
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
