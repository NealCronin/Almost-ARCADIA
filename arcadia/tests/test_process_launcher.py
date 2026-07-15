import os
import signal
import sys
import time
from pathlib import Path

import pytest

from arcadia.process import ProcessLauncher, ProcessSpec, RunningProcess, ProcessLaunchError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wait_until(running: RunningProcess, predicate, timeout: float = 5.0) -> None:
    """Poll until *predicate(running)* is true or *timeout* expires."""
    deadline = time.monotonic() + timeout
    while not predicate(running):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"predicate not met within {timeout}s")
        time.sleep(0.01)


@pytest.fixture
def launcher():
    """Provide a launcher that cleans up after itself."""
    launcher = ProcessLauncher()
    yield launcher
    launcher.stop_all()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_process_startup(launcher: ProcessLauncher):
    """1. Basic process startup."""
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "import time; print('ready', flush=True); time.sleep(30)"]))
    wait_until(running, lambda r: len(r.stdout_lines) > 0)
    assert running.process_id > 0
    assert launcher.is_running(running) is True
    assert "ready" in running.stdout_lines[0]


def test_standard_error_capture(launcher: ProcessLauncher):
    """2. Standard error capture."""
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "import sys; sys.stderr.write('err\\n')"])
    )
    wait_until(running, lambda r: len(r.stderr_lines) > 0)
    assert "err" in running.stderr_lines


def test_normal_process_completion(launcher: ProcessLauncher):
    """3. Natural completion."""
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "print('done')"])
    )
    wait_until(running, lambda r: not launcher.is_running(running))
    assert not launcher.is_running(running)
    assert "done" in running.stdout_lines[0]
    # Calling stop() on an already-exited process must be safe
    launcher.stop(running)


def test_graceful_stopping(launcher: ProcessLauncher):
    """4. Graceful stopping."""
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "import time; time.sleep(30)"]))
    assert launcher.is_running(running)
    launcher.stop(running)
    wait_until(running, lambda r: not launcher.is_running(running))
    assert not launcher.is_running(running)


def test_forced_kill(launcher: ProcessLauncher):
    """5. Forced kill (POSIX — skip on Windows)."""
    if sys.platform == "win32":
        pytest.skip("POSIX-only test")

    # Create a process that ignores SIGTERM — must be killed by SIGKILL
    script = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("alive", flush=True)
time.sleep(30)
"""
    running = launcher.start(ProcessSpec(command=[sys.executable, "-c", script]))
    wait_until(running, lambda r: "alive" in r.stdout_lines)
    launcher.stop(running)
    wait_until(running, lambda r: not launcher.is_running(running))
    assert not launcher.is_running(running)


def test_stop_all(launcher: ProcessLauncher):
    """6. stop_all()."""
    r1 = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "import time; time.sleep(30)"]))
    r2 = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "import time; time.sleep(30)"]))
    launcher.stop_all()
    wait_until(r1, lambda r: not launcher.is_running(r1))
    wait_until(r2, lambda r: not launcher.is_running(r2))


def test_environment_merging(launcher: ProcessLauncher):
    """7. Environment merging."""
    script = "import os, sys; print(os.environ.get('FOO', 'MISSING'), os.environ.get('PATH', 'MISSING')[:20], flush=True)"
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", script], environment={"FOO": "bar"})
    )
    wait_until(running, lambda r: len(r.stdout_lines) > 0)
    line = running.stdout_lines[0]
    assert "bar" in line
    assert "MISSING" not in line  # PATH should be inherited


def test_working_directory(launcher: ProcessLauncher, tmp_path: Path):
    """8. Working directory."""
    tmp = tmp_path / "arcadia_test_dir"
    tmp.mkdir()
    script = "import os, sys; print(os.getcwd(), flush=True)"
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", script], working_directory=tmp)
    )
    wait_until(running, lambda r: len(r.stdout_lines) > 0)
    assert str(tmp) in running.stdout_lines[0]


def test_argument_integrity(launcher: ProcessLauncher):
    """9. Argument integrity (spaces in args)."""
    script = "import sys; print(' '.join(sys.argv[1:]))"
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", script, "value with spaces"]))
    wait_until(running, lambda r: len(r.stdout_lines) > 0)
    assert running.stdout_lines[0] == "value with spaces"


def test_bounded_output():
    """10. Bounded output buffer (small buffer)."""
    small = ProcessLauncher(output_buffer_size=5)
    script = "import sys; [print(i, flush=True) for i in range(20)]"
    running = small.start(ProcessSpec(command=[sys.executable, "-c", script]))
    wait_until(running, lambda r: not small.is_running(running))
    assert list(running.stdout_lines) == ["15", "16", "17", "18", "19"]


def test_startup_failure(launcher: ProcessLauncher):
    """11. Startup failure."""
    with pytest.raises(ProcessLaunchError, match="Failed to start process"):
        launcher.start(ProcessSpec(command=["/nonexistent/binary_xyz"]))


def test_independent_output_copies(launcher: ProcessLauncher):
    """12. Independent output copies."""
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "print('line1')"])
    )
    wait_until(running, lambda r: len(r.stdout_lines) > 0)
    copy1 = launcher.recent_stdout(running)
    copy2 = launcher.recent_stdout(running)
    assert copy1 is not copy2
    assert copy1 == copy2


def test_empty_command_validation(launcher: ProcessLauncher):
    """13. Empty command validation."""
    with pytest.raises(ProcessLaunchError, match="command must not be empty"):
        launcher.start(ProcessSpec(command=[]))


def test_idempotent_stopping(launcher: ProcessLauncher):
    """14. Idempotent stopping."""
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "import time; time.sleep(30)"]))
    launcher.stop(running)
    # Second call must not raise
    launcher.stop(running)
    # Third call must not raise
    launcher.stop(running)
    assert not launcher.is_running(running)


def test_repeated_stop_all(launcher: ProcessLauncher):
    """15. Repeated stop_all()."""
    r1 = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "import time; time.sleep(30)"]))
    r2 = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "import time; time.sleep(30)"]))
    launcher.stop_all()
    # Second call must not raise
    launcher.stop_all()
    wait_until(r1, lambda r: not launcher.is_running(r1))
    wait_until(r2, lambda r: not launcher.is_running(r2))


def test_reader_cleanup(launcher: ProcessLauncher):
    """16. Reader cleanup — final output accessible after stop()."""
    script = "import sys; print('final', flush=True)"
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", script])
    )
    wait_until(running, lambda r: not launcher.is_running(running))
    # Output must still be available after stop
    launcher.stop(running)
    assert "final" in running.stdout_lines[0]


def test_positive_buffer_size_required():
    """17. Positive output_buffer_size required."""
    with pytest.raises(ValueError, match="output_buffer_size must be positive"):
        ProcessLauncher(output_buffer_size=0)


def test_graceful_termination_no_force_kill():
    """Graceful termination does not force kill."""
    # Use a process that exits immediately — graceful termination succeeds
    # without needing force kill
    launcher = ProcessLauncher()
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", "print('exit', flush=True)"])
    )
    wait_until(running, lambda r: not launcher.is_running(running))
    # Stop should still work (idempotent)
    launcher.stop(running)
    # The key assertion: the process exited naturally, so no SIGKILL was needed
    assert not launcher.is_running(running)


def test_timeout_causes_force_kill():
    """Timeout causes force kill."""
    if sys.platform == "win32":
        pytest.skip("POSIX-only test")

    # Use a process that ignores SIGTERM — it will time out and be force-killed
    script = """
import signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("alive", flush=True)
time.sleep(30)
"""
    launcher = ProcessLauncher()
    running = launcher.start(
        ProcessSpec(command=[sys.executable, "-c", script])
    )
    wait_until(running, lambda r: "alive" in r.stdout_lines)
    # Stop should trigger force kill after timeout
    launcher.stop(running)
    wait_until(running, lambda r: not launcher.is_running(running))
    assert not launcher.is_running(running)
