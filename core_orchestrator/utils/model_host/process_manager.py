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
    Registry + lifecycle manager for background child processes.

    Thread-safe – all public methods acquire an internal lock.
    """

    _instance: Optional["ProcessManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._registry: dict[str, _ProcessRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "ProcessManager":
        """Return the global singleton."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Force-recreate the singleton (useful for testing)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def start(
        self,
        name: str,
        cmd: list[str],
        capture_output: bool = True,
        **kwargs,
    ) -> _ProcessRecord:
        """
        Spawn a child process and register it.

        Parameters
        ----------
        name : str
            Human-readable identifier (also used as the dict key).
        cmd : list[str]
            Executable + arguments.
        capture_output : bool
            If ``True``, pipe stdout/stderr into in-memory buffers.
        **kwargs
            Forwarded to ``subprocess.Popen`` (env, cwd, …).
        """
        with self._lock:
            if name in self._registry:
                raise ValueError(f"Process '{name}' already registered")

            if capture_output:
                kwargs.setdefault("stdout", subprocess.PIPE)
                kwargs.setdefault("stderr", subprocess.PIPE)

            proc = subprocess.Popen(cmd, **kwargs)
            record = _ProcessRecord(pid=proc.pid, process=proc, name=name)
            self._registry[name] = record
            logger.info("Launched process '%s' (pid=%d) cmd=%s", name, proc.pid, cmd)
            return record

    def get(self, name: str) -> Optional[_ProcessRecord]:
        with self._lock:
            return self._registry.get(name)

    def list_all(self) -> list[_ProcessRecord]:
        with self._lock:
            return list(self._registry.values())

    def is_running(self, name: str) -> bool:
        """Return ``True`` if the process is alive and registered."""
        record = self.get(name)
        if record is None:
            return False
        # Poll updates the ``returncode`` attribute
        record.process.poll()
        record.alive = record.process.returncode is None
        if not record.alive:
            self._drain(record)
        return record.alive

    def read_output(self, name: str) -> tuple[str, str]:
        """Return captured stdout / stderr strings for *name*."""
        record = self.get(name)
        if record is None:
            return "", ""
        return "\n".join(record.stdout_buf), "\n".join(record.stderr_buf)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def stop(self, name: str, force: bool = False) -> bool:
        """
        Gracefully stop a registered process.

        Parameters
        ----------
        name : str
        force : bool
            If ``True``, call ``kill()`` instead of ``terminate()``.

        Returns
        -------
        bool
            ``True`` if the process was stopped, ``False`` if not found.
        """
        record = self.get(name)
        if record is None:
            return False

        if not record.alive:
            self.unregister(name)
            return True

        method = record.process.kill if force else record.process.terminate
        method()
        # Brief wait for SIGTERM to take effect
        if not force:
            import time

            deadline = time.monotonic() + 3
            while record.process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            # If still alive after 3 s, fall back to kill
            if record.process.poll() is None:
                logger.warning("Process '%s' did not terminate; forcing kill", name)
                record.process.kill()

        self._drain(record)
        self.unregister(name)
        logger.info("Stopped process '%s' (force=%s)", name, force)
        return True

    def stop_all(self, force: bool = False) -> None:
        """Stop every registered process."""
        names = list(self._registry.keys())
        for name in names:
            self.stop(name, force=force)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._registry.pop(name, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _drain(record: _ProcessRecord) -> None:
        """Drain pipe buffers into the record's in-memory lists."""
        proc = record.process
        if proc.stdout:
            try:
                out = proc.stdout.read()
                if out:
                    record.stdout_buf.append(out.decode("utf-8", errors="replace"))
            except Exception:
                pass
        if proc.stderr:
            try:
                err = proc.stderr.read()
                if err:
                    record.stderr_buf.append(err.decode("utf-8", errors="replace"))
            except Exception:
                pass
