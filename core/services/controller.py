from __future__ import annotations

import atexit
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .specs import ServiceEndpoint, ServiceSpec


@dataclass(slots=True)
class RunningService:
    spec: ServiceSpec
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: IO[str]


class ServiceController:
    """Starts, replaces, and stops services owned by this process."""

    def __init__(self, public_host: str = "127.0.0.1", log_dir: str = "logs") -> None:
        self.public_host = public_host
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._services: dict[int, RunningService] = {}
        self._lock = threading.RLock()
        atexit.register(self.stop_all)

    def start(self, spec: ServiceSpec) -> ServiceEndpoint:
        with self._lock:
            if spec.port in self._services:
                self.stop(spec.port)

            command = self._build_command(spec)
            log_path = self.log_dir / f"{spec.service_type}-{spec.port}.log"
            log_handle = log_path.open("a", encoding="utf-8")

            try:
                process = subprocess.Popen(
                    command,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            except Exception:
                log_handle.close()
                raise

            self._services[spec.port] = RunningService(
                spec=spec,
                process=process,
                log_path=log_path,
                log_handle=log_handle,
            )

            return ServiceEndpoint(
                host=self.public_host,
                port=spec.port,
                service_type=spec.service_type,
            )

    def stop(self, port: int) -> None:
        with self._lock:
            running = self._services.pop(port, None)
            if running is None:
                return

            process = running.process
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            finally:
                running.log_handle.close()

    def stop_all(self) -> None:
        with self._lock:
            for port in list(self._services):
                self.stop(port)

    def is_running(self, port: int) -> bool:
        with self._lock:
            running = self._services.get(port)
            if running is None:
                return False
            if running.process.poll() is not None:
                self._services.pop(port, None)
                running.log_handle.close()
                return False
            return True

    def list_services(self) -> list[dict[str, object]]:
        with self._lock:
            result: list[dict[str, object]] = []
            for port, running in list(self._services.items()):
                result.append(
                    {
                        "port": port,
                        "service_type": running.spec.service_type,
                        "running": self.is_running(port),
                        "settings": dict(running.spec.settings),
                        "log_path": str(running.log_path),
                    }
                )
            return result

    def _build_command(self, spec: ServiceSpec) -> list[str]:
        settings = spec.settings

        if "command" in settings:
            command = settings["command"]
            if not isinstance(command, list) or not all(
                isinstance(item, str) for item in command
            ):
                raise ValueError("'command' must be a list of strings.")
            return list(command)

        if spec.service_type == "llm":
            executable = str(settings.get("executable", "llama-server"))
            command = [
                executable,
                "--host",
                str(settings.get("bind_host", "0.0.0.0")),
                "--port",
                str(spec.port),
            ]

            hf_repo = settings.get("hf_repo")
            hf_file = settings.get("hf_file")
            model_path = settings.get("model_path")

            if hf_repo and hf_file:
                command.extend(["--hf-repo", str(hf_repo), "--hf-file", str(hf_file)])
            elif model_path:
                command.extend(["--model", str(model_path)])
            else:
                raise ValueError(
                    "LLM service requires either hf_repo + hf_file or model_path."
                )

            extra_args = settings.get("extra_args", [])
            if not isinstance(extra_args, list):
                raise ValueError("'extra_args' must be a list.")
            command.extend(str(item) for item in extra_args)
            return command

        if spec.service_type == "sam3":
            script = settings.get("script")
            checkpoint = settings.get("checkpoint")
            if not script or not checkpoint:
                raise ValueError("SAM3 service requires script and checkpoint settings.")

            executable = str(settings.get("python_executable", "python"))
            command = [
                executable,
                str(script),
                "--host",
                str(settings.get("bind_host", "0.0.0.0")),
                "--port",
                str(spec.port),
                "--checkpoint",
                str(checkpoint),
            ]
            extra_args = settings.get("extra_args", [])
            if not isinstance(extra_args, list):
                raise ValueError("'extra_args' must be a list.")
            command.extend(str(item) for item in extra_args)
            return command

        raise ValueError(f"Unsupported service type: {spec.service_type}")
