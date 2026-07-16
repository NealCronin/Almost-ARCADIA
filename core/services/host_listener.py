from __future__ import annotations

import ipaddress
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Callable, Literal

import requests

from core.config import HostListenerConfig
from core.errors import ServiceError
from core.networking import local_ipv4_addresses

HostListenerState = Literal["stopped", "starting", "running", "restarting", "rollback", "failed"]


class HostListenerError(ServiceError):
    """The Django-owned instruction listener could not be operated safely."""


class HostListenerRestartError(HostListenerError):
    def __init__(self, message: str, *, rollback_succeeded: bool | None) -> None:
        super().__init__(message)
        self.rollback_succeeded = rollback_succeeded


@dataclass(slots=True)
class HostListenerStatus:
    state: HostListenerState = "stopped"
    host: str = "127.0.0.1"
    port: int = 9000
    pid: int | None = None
    started_at: datetime | None = None
    message: str = "Instruction server is stopped"
    last_error: str | None = None

    @property
    def uptime_seconds(self) -> int | None:
        if self.started_at is None or self.state != "running":
            return None
        return max(0, int((datetime.now(timezone.utc) - self.started_at).total_seconds()))

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "state": self.state,
            "host": self.host,
            "port": self.port,
            "pid": self.pid,
            "uptime_seconds": self.uptime_seconds,
            "message": self.message,
            "health_url": self.health_url,
            "last_error": self.last_error,
        }


class HostListenerController:
    """Own the one instruction-server subprocess launched by this Django runtime."""

    def __init__(
        self,
        log_dir: str | Path = "logs/instruction",
        *,
        startup_timeout: float = 10.0,
        stop_timeout: float = 10.0,
        poll_interval: float = 0.1,
        local_addresses: Callable[[], set[str]] | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.startup_timeout = startup_timeout
        self.stop_timeout = stop_timeout
        self.poll_interval = poll_interval
        self._local_addresses = local_addresses or local_ipv4_addresses
        self._lock = threading.RLock()
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: IO[str] | None = None
        self._config: HostListenerConfig | None = None
        self._status = HostListenerStatus()

    def ensure_started(self, config: HostListenerConfig) -> HostListenerStatus:
        with self._lock:
            self._refresh_dead_process_locked()
            if self._is_running_locked() and self._config == config:
                return self.status()
            if self._is_running_locked():
                return self.restart(config)
            return self._start_locked(config)

    def start(self, config: HostListenerConfig) -> HostListenerStatus:
        with self._lock:
            self._refresh_dead_process_locked()
            if self._is_running_locked():
                raise HostListenerError("The Django-owned instruction server is already running.")
            return self._start_locked(config)

    def restart(
        self, config: HostListenerConfig, *, rollback_config: HostListenerConfig | None = None
    ) -> HostListenerStatus:
        with self._lock:
            self._validate_local_config(config)
            previous = rollback_config or self._config
            self._status = HostListenerStatus(
                state="restarting",
                host=config.host,
                port=config.port,
                message=f"Restarting instruction server on {config.host}:{config.port}",
            )
            self._stop_locked()
            try:
                return self._start_locked(config)
            except HostListenerError as exc:
                if previous is None:
                    raise HostListenerRestartError(str(exc), rollback_succeeded=None) from exc
                self._status = HostListenerStatus(
                    state="rollback",
                    host=previous.host,
                    port=previous.port,
                    message="Replacement failed; restoring the previous instruction server",
                    last_error=str(exc),
                )
                try:
                    restored = self._start_locked(previous, last_error=str(exc))
                except HostListenerError as rollback_exc:
                    message = f"Replacement failed: {exc}. Rollback failed: {rollback_exc}"
                    self._status = HostListenerStatus(
                        state="failed",
                        host=previous.host,
                        port=previous.port,
                        message="Instruction server replacement and rollback failed",
                        last_error=message,
                    )
                    raise HostListenerRestartError(message, rollback_succeeded=False) from rollback_exc
                restored.message = "Replacement failed; previous instruction server was restored"
                restored.last_error = str(exc)
                raise HostListenerRestartError(
                    f"Replacement failed: {exc}. Previous instruction server was restored.",
                    rollback_succeeded=True,
                ) from exc

    def stop(self) -> HostListenerStatus:
        with self._lock:
            self._stop_locked()
            return self.status()

    def close(self) -> None:
        self.stop()

    def status(self) -> HostListenerStatus:
        with self._lock:
            self._refresh_dead_process_locked()
            return HostListenerStatus(
                state=self._status.state,
                host=self._status.host,
                port=self._status.port,
                pid=self._status.pid,
                started_at=self._status.started_at,
                message=self._status.message,
                last_error=self._status.last_error,
            )

    def logs(self, tail: int = 200) -> str:
        path = self.log_dir / "instruction-server.log"
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except FileNotFoundError:
            return ""
        except OSError as exc:
            raise HostListenerError(f"Could not read instruction server log: {exc}") from exc
        return "\n".join(lines[-max(1, min(tail, 5000)) :])

    def _start_locked(self, config: HostListenerConfig, *, last_error: str | None = None) -> HostListenerStatus:
        self._status = HostListenerStatus(
            state="starting",
            host=config.host,
            port=config.port,
            message=f"Starting instruction server on {config.host}:{config.port}",
            last_error=last_error,
        )
        log_handle: IO[str] | None = None
        try:
            self._validate_local_config(config)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / "instruction-server.log"
            log_handle = log_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                [
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
                ],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
        except HostListenerError:
            self._status.state = "failed"
            self._status.message = "Instruction server configuration is invalid"
            raise
        except OSError as exc:
            if log_handle is not None:
                log_handle.close()
            self._status.state = "failed"
            self._status.message = "Instruction server could not be launched"
            self._status.last_error = str(exc)
            raise HostListenerError(f"Could not launch instruction server: {exc}") from exc
        self._process = process
        self._log_handle = log_handle
        self._config = config
        self._status.pid = process.pid
        try:
            self._wait_ready_locked()
        except HostListenerError as exc:
            self._stop_locked()
            self._status.state = "failed"
            self._status.message = "Instruction server failed to start"
            self._status.last_error = str(exc)
            raise
        self._status = HostListenerStatus(
            state="running",
            host=config.host,
            port=config.port,
            pid=process.pid,
            started_at=datetime.now(timezone.utc),
            message="Instruction server is running",
            last_error=last_error,
        )
        return self._status

    def _wait_ready_locked(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        url = self._status.health_url
        last_error = "health endpoint did not become ready"
        while True:
            process = self._process
            if process is None:
                raise HostListenerError(f"Instruction server startup lost its child before {url} became ready.")
            if process.poll() is not None:
                raise HostListenerError(
                    f"Instruction server exited with code {process.returncode} before {url} became ready."
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HostListenerError(f"Timed out waiting for instruction server health at {url}: {last_error}")
            try:
                response = requests.get(url, timeout=min(1.0, max(0.1, remaining)))
                if response.status_code == 200 and response.json() == {"status": "ok", "service": "instruction"}:
                    return
                last_error = f"health returned HTTP {response.status_code}"
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
            time.sleep(min(self.poll_interval, remaining))

    def _stop_locked(self) -> None:
        process = self._process
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=self.stop_timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=self.stop_timeout)
        finally:
            self._process = None
            self._config = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            self._status = HostListenerStatus(
                state="stopped",
                host=self._status.host,
                port=self._status.port,
                message="Instruction server is stopped",
                last_error=self._status.last_error,
            )

    def _refresh_dead_process_locked(self) -> None:
        if self._process is None or self._process.poll() is None:
            return
        returncode = self._process.returncode
        self._process = None
        self._config = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self._status.state = "failed"
        self._status.pid = None
        self._status.message = "Instruction server exited unexpectedly"
        self._status.last_error = f"Instruction server exited with code {returncode}."

    def _is_running_locked(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _validate_local_config(self, config: HostListenerConfig) -> None:
        try:
            address = ipaddress.ip_address(config.host)
        except ValueError as exc:
            raise HostListenerError(f"Invalid instruction-server IP address: {config.host!r}") from exc
        if address.version != 4:
            raise HostListenerError("Instruction server currently supports IPv4 addresses only.")
        if str(address) not in self._local_addresses():
            raise HostListenerError(
                f"Instruction-server IP {config.host} is not assigned to a local network interface."
            )
