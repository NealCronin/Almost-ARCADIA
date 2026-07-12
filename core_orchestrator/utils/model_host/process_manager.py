"""
process_manager.py

Tracks local background child-process lifecycles via ``subprocess.Popen``.

Key guarantees
--------------
* **Spawn race fixed**: ``start()`` holds the lock for the *entire*
  spawn cycle.  A second caller for the same name receives an
  ``AlreadyRunningError`` instead of spawning a duplicate.
* **No implicit replacement**: ``start()`` does NOT silently stop an
  existing process.  Callers must call ``stop()`` explicitly.
* ``process.poll()`` is the source of truth for alive status.
* Bounded stdout/stderr buffers (``deque(maxlen=2000)``).
* Reader-thread EOF does not mark the process dead.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

MAX_LOG_LINES = 2000


class AlreadyRunningError(Exception):
    """Raised when start() is called for a name that is already tracked."""


class _ProcessRecord:
    __slots__ = ("pid", "process", "name", "stdout_buf", "stderr_buf", "alive", "_exited")

    def __init__(self, pid: int, process: subprocess.Popen, name: str) -> None:
        self.pid = pid
        self.process = process
        self.name = name
        self.stdout_buf: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.stderr_buf: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.alive = True
        self._exited = False  # set when poll() returns non-None

    @property
    def exited(self) -> bool:
        if self._exited:
            return True
        if self.process.poll() is not None:
            self._exited = True
            self.alive = False
            return True
        return False

    def __repr__(self) -> str:
        return f"<ProcessRecord {self.name} pid={self.pid} alive={self.alive}>"


class ProcessManager:
    """
    Thread-safe process lifecycle manager.

    ``start()`` raises ``AlreadyRunningError`` if a process with the same
    name is currently tracked.  This prevents the spawn race where two
    callers both pass an initial check, spawn two processes, and one
    becomes untracked.
    """

    _instance: Optional["ProcessManager"] = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._processes: dict[str, _ProcessRecord] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ProcessManager":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start(
        self,
        name: str,
        cmd: list[str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
    ) -> int:
        """
        Start a background process and track it by *name*.

        Raises ``AlreadyRunningError`` if a process with *name* is
        already tracked and still alive.

        Returns the PID of the spawned child.
        """
        with self._lock:
            if name in self._processes:
                record = self._processes[name]
                if not record.exited:
                    raise AlreadyRunningError(
                        f"Process '{name}' is already running (PID {record.pid})"
                    )
                # Remove dead record
                del self._processes[name]

            # Spawn while holding the lock so no other thread can enter
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                env=env,
                bufsize=1,
            )

            record = _ProcessRecord(proc.pid, proc, name)

            t_out = threading.Thread(
                target=self._read_stream, args=(record, record.stdout_buf, proc.stdout)
            )
            t_err = threading.Thread(
                target=self._read_stream, args=(record, record.stderr_buf, proc.stderr)
            )
            t_out.daemon = True
            t_err.daemon = True
            t_out.start()
            t_err.start()

            self._processes[name] = record

            logger.info("Started process %s (PID %d): %s", name, proc.pid, " ".join(cmd))
            return proc.pid

    def _read_stream(
        self, record: _ProcessRecord, buf: deque[str], stream: Optional[object]
    ) -> None:
        if stream is None:
            return
        try:
            for line in stream:
                buf.append(line.rstrip("\n"))
                logger.debug("[%s] %s", record.name, line.rstrip())
        except Exception:
            logger.exception("Error reading stream for %s", record.name)
        # DO NOT set record.alive = False here.
        # Use record.exited / process.poll() as source of truth.

    def is_running(self, name: str) -> bool:
        with self._lock:
            if name not in self._processes:
                return False
            record = self._processes[name]
            if record.exited:
                return False
            return True

    def get_pid(self, name: str) -> Optional[int]:
        with self._lock:
            if name not in self._processes:
                return None
            return self._processes[name].pid

    def stop(self, name: str, force: bool = False) -> bool:
        """Gracefully terminate (SIGTERM) or force-kill (SIGKILL)."""
        with self._lock:
            return self._stop_no_lock(name, force=force)

    def _stop_no_lock(self, name: str, force: bool) -> bool:
        if name not in self._processes:
            return False

        record = self._processes[name]
        if record.exited:
            del self._processes[name]
            return True

        try:
            if force:
                logger.info("Killing process %s (PID %d)", name, record.pid)
                record.process.kill()
            else:
                logger.info("Terminating process %s (PID %d)", name, record.pid)
                record.process.terminate()

            try:
                record.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process %s did not terminate, killing", name)
                record.process.kill()
                record.process.wait()

            record.alive = False
            record._exited = True
            del self._processes[name]
            return True
        except Exception:
            logger.exception("Error stopping process %s", name)
            return False

    def kill(self, name: str) -> bool:
        return self.stop(name, force=True)

    def get_output(self, name: str, clear: bool = False) -> tuple[list[str], list[str]]:
        """Return (stdout_lines, stderr_lines) for a named process."""
        with self._lock:
            if name not in self._processes:
                return ([], [])
            record = self._processes[name]
            stdout = list(record.stdout_buf)
            stderr = list(record.stderr_buf)
            if clear:
                record.stdout_buf.clear()
                record.stderr_buf.clear()
            return (stdout, stderr)

    def list_processes(self) -> list[dict]:
        with self._lock:
            result = []
            for name, record in self._processes.items():
                result.append(
                    {
                        "name": name,
                        "pid": record.pid,
                        "alive": record.alive and not record.exited,
                        "returncode": record.process.returncode,
                    }
                )
            return result


def reset_process_manager_for_tests() -> None:
    """Reset the global ProcessManager singleton. Intended for test isolation."""
    global _pm
    ProcessManager._instance = None