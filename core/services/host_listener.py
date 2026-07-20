from __future__ import annotations

import atexit
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import requests

from core.config import HostListenerConfig
from core.errors import ArcadiaError
from core.networking import local_ipv4_addresses


class HostListenerError(ArcadiaError):
    pass


class HostListenerRestartError(HostListenerError):
    pass


@dataclass(frozen=True, slots=True)
class HostListenerStatus:
    running: bool
    host: str
    port: int
    pid: int | None = None
    uptime_seconds: float | None = None
    last_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "running": self.running,
            "host": self.host,
            "port": self.port,
            "pid": self.pid,
            "uptime_seconds": self.uptime_seconds,
            "last_error": self.last_error,
        }


class HostListenerManager:
    """Own one instruction-server subprocess started by Django."""

    def __init__(self, *, log_dir: str | Path = "logs/instruction", startup_timeout: float = 15.0) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.startup_timeout = startup_timeout
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: IO[str] | None = None
        self._config = HostListenerConfig()
        self._started_at: float | None = None
        self._last_error: str | None = None
        atexit.register(self.stop)

    def _validate_local_host(self, config: HostListenerConfig) -> None:
        if config.host not in local_ipv4_addresses():
            raise HostListenerError(f"Instruction IP {config.host} is not assigned to this computer.")

    def _command(self, config: HostListenerConfig) -> list[str]:
        return [
            sys.executable,
            "-m",
            "core.services.instruction_server",
            "--host",
            config.host,
            "--public-host",
            config.host,
            "--port",
            str(config.port),
            "--log-dir",
            str(self.log_dir),
        ]

    def start(self, config: HostListenerConfig) -> None:
        with self._lock:
            self._validate_local_host(config)
            if self._process is not None and self._process.poll() is None:
                raise HostListenerError("The instruction server is already running.")
            log_path = self.log_dir / "listener.log"
            handle = log_path.open("a", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    self._command(config),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    shell=False,
                    env=os.environ.copy(),
                )
            except Exception:
                handle.close()
                raise
            self._process = process
            self._log_handle = handle
            self._config = config
            self._started_at = time.monotonic()
            self._last_error = None
            try:
                self._wait_ready(config)
            except Exception as exc:
                self._last_error = str(exc)
                self._stop_locked()
                raise HostListenerError(str(exc)) from exc

    def _wait_ready(self, config: HostListenerConfig) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "listener is starting"
        while time.monotonic() < deadline:
            if self._process is None or self._process.poll() is not None:
                code = None if self._process is None else self._process.returncode
                raise HostListenerError(f"Instruction server exited during startup with code {code}.")
            try:
                response = requests.get(f"http://{config.host}:{config.port}/health", timeout=1.0)
                if response.ok:
                    return
                last_error = f"health returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(0.2)
        raise HostListenerError(f"Instruction server did not become ready: {last_error}")

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._started_at = None

    def restart(self, replacement: HostListenerConfig, *, rollback_config: HostListenerConfig | None = None) -> None:
        with self._lock:
            prior = rollback_config or self._config
            self._stop_locked()
            try:
                self.start(replacement)
            except Exception as replacement_error:
                try:
                    self.start(prior)
                except Exception as rollback_error:
                    raise HostListenerRestartError(
                        f"Replacement failed: {replacement_error}. Rollback also failed: {rollback_error}"
                    ) from rollback_error
                raise HostListenerRestartError(
                    f"Replacement failed: {replacement_error}. The previous listener was restored."
                ) from replacement_error

    def status(self) -> HostListenerStatus:
        with self._lock:
            running = self._process is not None and self._process.poll() is None
            uptime = time.monotonic() - self._started_at if running and self._started_at is not None else None
            return HostListenerStatus(
                running=running,
                host=self._config.host,
                port=self._config.port,
                pid=self._process.pid if running and self._process is not None else None,
                uptime_seconds=uptime,
                last_error=self._last_error,
            )
