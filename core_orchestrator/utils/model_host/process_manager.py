"""
process_manager.py

Tracks local background child-process lifecycles via ``subprocess.Popen``.
Provides PID mapping, safe stdout/stderr capture, and graceful / forced
cleanup so zombie processes never accumulate.

Designed as a **singleton-registry** – call ``ProcessManager.instance()``
from anywhere in the app.
"""

import logging
import subprocess
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class _ProcessRecord:
    """Lightweight container for a tracked child process."""

    __slots__ = ("pid", "process", "name", "stdout_buf", "stderr_buf", "alive")

    def __init__(self, pid: int, process: subprocess.Popen, name: str) -> None:
        self.pid = pid
        self.process = process
        self.name = name
        self.stdout_buf: list[str] = []
        self.stderr_buf: list[str] = []
        self.alive = True

    def __repr__(self) -> str:  # noqa: D401
        return f"<ProcessRecord {self.name} pid={self.pid} alive={self.alive}>"


class ProcessManager:
    """
    Thread-safe process lifecycle manager.

    Features
    --------
    * Spawn background binaries with automatic stdout/stderr streaming.
    * Track processes by name → one process per name (latest spawn wins).
    * Safe ``terminate()`` (SIGTERM) and ``kill()`` (SIGKILL) methods.
    * In-memory stdout/stderr buffers for debugging.
    """

    _instance: Optional["ProcessManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._processes: dict[str, _ProcessRecord] = {}
        self._lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ProcessManager":
        """Return the singleton ProcessManager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ProcessManager()
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

        Returns the PID of the spawned child.
        """
        with self._lock:
            # Stop existing process with same name if any
            if name in self._processes:
                logger.warning("Process %s already running, stopping first", name)
                self._stop_no_lock(name, force=False)

        # Spawn the process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=env,
            bufsize=1,  # line buffered
        )

        record = _ProcessRecord(proc.pid, proc, name)

        # Start reader threads
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

        with self._lock:
            self._processes[name] = record

        logger.info("Started process %s (PID %d): %s", name, proc.pid, " ".join(cmd))
        return proc.pid

    def _read_stream(
        self, record: _ProcessRecord, buf: list[str], stream: Optional[object]
    ) -> None:
        """Thread target to read a stream into buffer."""
        if stream is None:
            return
        try:
            for line in stream:
                buf.append(line)
                logger.debug("[%s] %s", record.name, line.rstrip())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error reading stream for %s: %s", record.name, exc)
        finally:
            record.alive = False

    def is_running(self, name: str) -> bool:
        """Check if a named process is still alive."""
        with self._lock:
            if name not in self._processes:
                return False
            record = self._processes[name]
            if record.process.poll() is not None:
                record.alive = False
                return False
            return True

    def get_pid(self, name: str) -> Optional[int]:
        """Return the PID for a named process, or None."""
        with self._lock:
            if name not in self._processes:
                return None
            return self._processes[name].pid

    def stop(self, name: str, force: bool = False) -> bool:
        """
        Gracefully terminate a named process.

        If *force* is True, use SIGKILL instead of SIGTERM.
        Returns True if the process was found and terminated.
        """
        with self._lock:
            return self._stop_no_lock(name, force=force)

    def _stop_no_lock(self, name: str, force: bool) -> bool:
        """Internal stop without lock (caller must hold lock)."""
        if name not in self._processes:
            return False

        record = self._processes[name]
        if record.process.poll() is not None:
            # Already dead
            del self._processes[name]
            return True

        try:
            if force:
                logger.info("Killing process %s (PID %d)", name, record.pid)
                record.process.kill()
            else:
                logger.info("Terminating process %s (PID %d)", name, record.pid)
                record.process.terminate()

            # Wait briefly for graceful shutdown
            try:
                record.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Process %s did not terminate, killing", name)
                record.process.kill()
                record.process.wait()

            record.alive = False
            del self._processes[name]
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error stopping process %s: %s", name, exc)
            return False

    def kill(self, name: str) -> bool:
        """Force-kill a named process (SIGKILL). Returns True if found."""
        return self.stop(name, force=True)

    def get_output(self, name: str, clear: bool = False) -> tuple[list[str], list[str]]:
        """
        Get captured stdout/stderr buffers for a named process.

        If *clear* is True, empty the buffers after reading.
        """
        with self._lock:
            if name not in self._processes:
                return ([], [])
            record = self._processes[name]
            stdout = record.stdout_buf.copy()
            stderr = record.stderr_buf.copy()
            if clear:
                record.stdout_buf.clear()
                record.stderr_buf.clear()
            return (stdout, stderr)

    def list_processes(self) -> list[dict]:
        """Return a list of all tracked processes with status info."""
        with self._lock:
            result = []
            for name, record in self._processes.items():
                result.append(
                    {
                        "name": name,
                        "pid": record.pid,
                        "alive": record.alive and record.process.poll() is None,
                        "returncode": record.process.returncode,
                    }
                )
            return result
