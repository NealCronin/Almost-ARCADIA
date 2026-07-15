import logging
import os
import signal
import sys
import threading
import subprocess
try:
    from subprocess import CREATE_NEW_PROCESS_GROUP
except ImportError:
    CREATE_NEW_PROCESS_GROUP = 0

from .models import ProcessSpec, RunningProcess
logger = logging.getLogger(__name__)


TERMINATE_TIMEOUT_SECONDS: float = 3.0

class ProcessLaunchError(RuntimeError):
    """Raised when a child process fails to start."""


class ProcessLauncher:
    """Start, track, capture output from, and stop child processes.

    Completely unaware of AI models, inference services, or any
    application-specific logic.
    """

    def __init__(self, output_buffer_size: int = 200):
        self._processes: dict[int, RunningProcess] = {}
        self._lock = threading.Lock()
        self._output_buffer_size = output_buffer_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, spec: ProcessSpec) -> RunningProcess:
        """Start a child process and return a handle to it.

        Raises ``ProcessLaunchError`` if the command is empty or the
        process fails to start.
        """
        if not spec.command:
            raise ProcessLaunchError("command must not be empty")

        kwargs: dict = {
            "args": spec.command,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "shell": False,
        }

        if spec.working_directory is not None:
            kwargs["cwd"] = spec.working_directory

        if spec.environment is not None:
            env = os.environ.copy()
            env.update(spec.environment)
            kwargs["env"] = env
        # Platform-specific process isolation
        if sys.platform == "win32":
            kwargs["creationflags"] = CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(**kwargs)
        except Exception as exc:
            raise ProcessLaunchError(f"Failed to start process: {spec.command!r}") from exc

        running = RunningProcess(process=proc, spec=spec, output_buffer_size=self._output_buffer_size)
        with self._lock:
            self._processes[proc.pid] = running

        # Start background reader threads
        self._start_reader(running, proc.stdout, "stdout")
        self._start_reader(running, proc.stderr, "stderr")

        logger.info("Started process %s (pid=%d)", spec.command[0], proc.pid)
        return running

    def is_running(self, running: RunningProcess) -> bool:
        """Return ``True`` if the process has not yet exited."""
        return running.process.poll() is None

    def stop(self, running: RunningProcess) -> None:
        """Gracefully stop a process, then force-kill if needed.

        Idempotent: safe to call multiple times.
        """
        proc = running.process
        pid = proc.pid

        # Clean up tracking regardless of state
        with self._lock:
            self._processes.pop(pid, None)

        if proc.poll() is not None:
            return  # already exited — just cleaned up tracking

        # Graceful termination
        try:
            if sys.platform == "win32":
                proc.terminate()
            else:
                os.killpg(pid, signal.SIGTERM)
        except Exception:
            pass  # escalate to force kill below

        try:
            proc.wait(timeout=TERMINATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass  # still alive — force kill

        # Force kill
        try:
            if sys.platform == "win32":
                proc.kill()
            else:
                os.killpg(pid, signal.SIGKILL)
        except Exception:
            pass

        try:
            proc.wait(timeout=1)
        except Exception:
            pass

        logger.info("Stopped process %s (pid=%d)", running.spec.command[0], pid)

    def stop_all(self) -> None:
        """Stop every tracked process. Individual failures are suppressed."""
        with self._lock:
            snapshot = list(self._processes.values())

        for running in snapshot:
            try:
                self.stop(running)
            except Exception:
                logger.exception("Failed to stop process %s", running.spec.command[0])

    def recent_stdout(self, running: RunningProcess) -> list[str]:
        """Return a copy of the most recent stdout lines."""
        return list(running.stdout_lines)

    def recent_stderr(self, running: RunningProcess) -> list[str]:
        """Return a copy of the most recent stderr lines."""
        return list(running.stderr_lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _start_reader(self, running: RunningProcess, stream, name: str) -> None:
        """Background thread that reads lines from *stream* into the buffer."""

        def _read():
            try:
                for line in stream:
                    line = line.rstrip("\n\r")
                    if name == "stdout":
                        running.stdout_lines.append(line)
                    else:
                        running.stderr_lines.append(line)
            except Exception:
                logger.exception("Error reading %s for process %s", name, running.spec.command[0])
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
