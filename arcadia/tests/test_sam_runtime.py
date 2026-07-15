"""Tests for SAMRuntime.

Uses fake launcher and probe to test the runtime lifecycle without
launching a real process.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest import mock

import pytest

from arcadia.process import ProcessLauncher, ProcessSpec, RunningProcess
from arcadia.contracts import RunningService, ServiceEndpoint, ServiceSpec, ModelSpec
from arcadia.runtimes.sam import SAMRuntime, SAMRuntimeError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeRunningProcess:
    """A fake process that tracks state for testing, matching RunningProcess-like shape."""

    def __init__(self, command: list[str], should_exit: bool = False):
        self.command = command
        self._should_exit = should_exit
        self._running = True

    def is_running(self) -> bool:
        if self._should_exit:
            self._running = False
            return False
        return self._running

    def stop(self) -> None:
        self._running = False


class FakeProcessLauncher:
    """A fake launcher that returns controlled FakeRunningProcess instances."""

    def __init__(self):
        self._processes: list[FakeRunningProcess] = []
        self._stderr: str = ""

    def start(self, spec: ProcessSpec) -> FakeRunningProcess:
        proc = FakeRunningProcess(command=spec.command)
        self._processes.append(proc)
        return proc

    def is_running(self, process: FakeRunningProcess) -> bool:
        return process.is_running()

    def stop(self, process: FakeRunningProcess) -> None:
        process.stop()

    def recent_stderr(self, process: FakeRunningProcess) -> str:
        return self._stderr

    def add_stderr(self, text: str) -> None:
        self._stderr = text


class FakeReadinessProbe:
    """A fake readiness probe that returns a configurable value."""

    def __init__(self, return_value: bool = True, delay: float = 0.0):
        self._return_value = return_value
        self._delay = delay
        self._call_count = 0

    def __call__(self, host: str, port: int) -> bool:
        self._call_count += 1
        time.sleep(self._delay)
        return self._return_value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSAMRuntimeError:
    def test_is_subclass_of_runtime_error(self):
        assert issubclass(SAMRuntimeError, RuntimeError)


class TestSAMRuntimeInit:
    def test_stores_constructor_args(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe()
        runtime = SAMRuntime(
            process_launcher=launcher,
            python_executable=Path("/usr/bin/python3"),
            readiness_probe=probe,
            startup_timeout=60.0,
            poll_interval=0.1,
        )
        assert runtime._process_launcher is launcher
        assert runtime._python_executable == Path("/usr/bin/python3")
        assert runtime._readiness_probe is probe
        assert runtime._startup_timeout == 60.0
        assert runtime._poll_interval == 0.1

    def test_defaults(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        assert runtime._python_executable is not None
        assert runtime._readiness_probe is not None
        assert runtime._startup_timeout == 120.0
        assert runtime._poll_interval == 0.25


class TestResolveCheckpointPath:
    def test_returns_path_when_exists(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/test.pt"))
        Path("/tmp/test.pt").touch()
        result = runtime._resolve_checkpoint_path(model)
        assert result == Path("/tmp/test.pt")

    @mock.patch("pathlib.Path.exists", return_value=False)
    def test_raises_when_path_missing(self, mock_exists):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/nonexistent.pt"))
        with pytest.raises(SAMRuntimeError, match="not found"):
            runtime._resolve_checkpoint_path(model)

    def test_raises_when_local_path_none(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=None)
        with pytest.raises(SAMRuntimeError, match="local_path is required"):
            runtime._resolve_checkpoint_path(model)


class TestBuildProcessSpec:
    def test_basic_command_structure(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1"},
        )
        proc_spec = runtime._build_process_spec(spec, Path("/tmp/model.pt"))
        assert proc_spec.command[0] == str(runtime._python_executable)
        assert proc_spec.command[1] == "-m"
        assert "sam_server" in proc_spec.command[2]
        assert "--checkpoint" in proc_spec.command
        assert "--port" in proc_spec.command
        assert "8080" in proc_spec.command

    def test_device_setting(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1", "device": "cuda"},
        )
        proc_spec = runtime._build_process_spec(spec, Path("/tmp/model.pt"))
        assert "--device" in proc_spec.command
        assert "cuda" in proc_spec.command

    def test_half_precision_setting(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1", "half_precision": True},
        )
        proc_spec = runtime._build_process_spec(spec, Path("/tmp/model.pt"))
        assert "--half-precision" in proc_spec.command
        assert "true" in proc_spec.command

    def test_half_precision_false(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1", "half_precision": False},
        )
        proc_spec = runtime._build_process_spec(spec, Path("/tmp/model.pt"))
        assert "--half-precision" in proc_spec.command
        assert "false" in proc_spec.command

    def test_default_confidence_setting(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1", "default_confidence": 0.5},
        )
        proc_spec = runtime._build_process_spec(spec, Path("/tmp/model.pt"))
        assert "--default-confidence" in proc_spec.command
        assert "0.5" in proc_spec.command

    def test_extra_args(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1", "extra_args": ["--verbose", "--debug"]},
        )
        proc_spec = runtime._build_process_spec(spec, Path("/tmp/model.pt"))
        assert "--verbose" in proc_spec.command
        assert "--debug" in proc_spec.command

    def test_no_extra_args(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1"},
        )
        proc_spec = runtime._build_process_spec(spec, Path("/tmp/model.pt"))
        assert "--verbose" not in proc_spec.command
        assert "--debug" not in proc_spec.command

    def test_deterministic_order(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={
                "host": "127.0.0.1",
                "device": "cuda",
                "half_precision": True,
                "default_confidence": 0.5,
                "extra_args": ["--verbose"],
            },
        )
        proc_spec = runtime._build_process_spec(spec, Path("/tmp/model.pt"))
        idx_checkpoint = proc_spec.command.index("--checkpoint")
        idx_host = proc_spec.command.index("--host")
        idx_port = proc_spec.command.index("--port")
        idx_device = proc_spec.command.index("--device")
        idx_half = proc_spec.command.index("--half-precision")
        idx_conf = proc_spec.command.index("--default-confidence")
        assert idx_checkpoint < idx_host < idx_port < idx_device < idx_half < idx_conf


class TestValidate:
    def test_wrong_service_type(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="generation",
            model=model,
            port=8080,
            settings={},
        )
        with pytest.raises(SAMRuntimeError, match="segmentation"):
            runtime._validate(spec)

    def test_missing_model(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        spec = ServiceSpec(
            service_type="segmentation",
            model=None,
            port=8080,
            settings={},
        )
        with pytest.raises(SAMRuntimeError, match="Model is required"):
            runtime._validate(spec)

    def test_missing_local_path(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=None)
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={},
        )
        with pytest.raises(SAMRuntimeError, match="local_path is required"):
            runtime._validate(spec)

    def test_port_out_of_range(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=0,
            settings={},
        )
        with pytest.raises(SAMRuntimeError, match="Port must be 1-65535"):
            runtime._validate(spec)

    def test_port_too_high(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=70000,
            settings={},
        )
        with pytest.raises(SAMRuntimeError, match="Port must be 1-65535"):
            runtime._validate(spec)

    def test_bad_host_type(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": 123},
        )
        with pytest.raises(SAMRuntimeError, match="host must be a non-empty string"):
            runtime._validate(spec)

    def test_empty_host(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": ""},
        )
        with pytest.raises(SAMRuntimeError, match="host must be a non-empty string"):
            runtime._validate(spec)

    def test_bad_device_type(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"device": 123},
        )
        with pytest.raises(SAMRuntimeError, match="device must be a non-empty string"):
            runtime._validate(spec)

    def test_bad_half_precision_type(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"half_precision": "true"},
        )
        with pytest.raises(SAMRuntimeError, match="half_precision must be a bool"):
            runtime._validate(spec)

    def test_bad_confidence_type(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"default_confidence": "0.5"},
        )
        with pytest.raises(SAMRuntimeError, match="default_confidence must be a number"):
            runtime._validate(spec)

    def test_confidence_out_of_range(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"default_confidence": 1.5},
        )
        with pytest.raises(SAMRuntimeError, match="default_confidence must be between 0.0 and 1.0"):
            runtime._validate(spec)

    def test_bad_extra_args_type(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"extra_args": ["a", 1]},
        )
        with pytest.raises(SAMRuntimeError, match="extra_args items must be strings"):
            runtime._validate(spec)

    def test_unknown_setting(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"unknown_key": "value"},
        )
        with pytest.raises(SAMRuntimeError, match="Unknown setting: unknown_key"):
            runtime._validate(spec)

    def test_valid_spec(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={
                "host": "127.0.0.1",
                "device": "cpu",
                "half_precision": False,
                "default_confidence": 0.25,
                "extra_args": [],
            },
        )
        runtime._validate(spec)  # should not raise


class TestDefaultReadinessProbe:
    def test_returns_true_when_ready(self):
        probe = SAMRuntime._default_readiness_probe
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_context = mock.MagicMock()
            mock_context.__enter__.return_value.read.return_value = (
                b'{"status": "ready", "service_type": "segmentation"}'
            )
            mock_urlopen.return_value = mock_context
            assert probe("127.0.0.1", 8080) is True

    def test_returns_false_when_not_ready(self):
        probe = SAMRuntime._default_readiness_probe
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_context = mock.MagicMock()
            mock_context.__enter__.return_value.read.return_value = (
                b'{"status": "loading", "service_type": "segmentation"}'
            )
            mock_urlopen.return_value = mock_context
            assert probe("127.0.0.1", 8080) is False

    def test_returns_false_on_exception(self):
        probe = SAMRuntime._default_readiness_probe
        with mock.patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
            assert probe("127.0.0.1", 8080) is False


class TestWaitForService:
    def test_returns_immediately_when_ready(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=True)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=1.0,
            poll_interval=0.01,
        )
        proc = FakeRunningProcess(command=["test"])
        runtime._wait_for_service(proc, "127.0.0.1", 8080)
        assert probe._call_count == 1

    def test_raises_on_timeout(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=False)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=0.1,
            poll_interval=0.01,
        )
        proc = FakeRunningProcess(command=["test"])
        with pytest.raises(SAMRuntimeError, match="did not become ready"):
            runtime._wait_for_service(proc, "127.0.0.1", 8080)

    def test_raises_on_early_exit(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=False)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=1.0,
            poll_interval=0.01,
        )
        proc = FakeRunningProcess(command=["test"], should_exit=True)
        launcher.add_stderr("model not found")
        with pytest.raises(SAMRuntimeError, match="Process exited unexpectedly"):
            runtime._wait_for_service(proc, "127.0.0.1", 8080)


class TestStart:
    def test_returns_running_service(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=True)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=1.0,
            poll_interval=0.01,
        )
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1"},
        )
        service = runtime.start(spec)
        assert isinstance(service, RunningService)
        assert service.endpoint.service_type == "segmentation"
        assert service.endpoint.port == 8080

    def test_calls_validate(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=True)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=1.0,
            poll_interval=0.01,
        )
        with mock.patch.object(runtime, "_validate") as mock_validate:
            model = ModelSpec(local_path=Path("/tmp/model.pt"))
            Path("/tmp/model.pt").touch()
            spec = ServiceSpec(
                service_type="segmentation",
                model=model,
                port=8080,
                settings={"host": "127.0.0.1"},
            )
            runtime.start(spec)
            mock_validate.assert_called_once_with(spec)

    def test_calls_resolve_checkpoint_path(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=True)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=1.0,
            poll_interval=0.01,
        )
        with mock.patch.object(runtime, "_resolve_checkpoint_path") as mock_resolve:
            mock_resolve.return_value = Path("/tmp/model.pt")
            model = ModelSpec(local_path=Path("/tmp/model.pt"))
            Path("/tmp/model.pt").touch()
            spec = ServiceSpec(
                service_type="segmentation",
                model=model,
                port=8080,
                settings={"host": "127.0.0.1"},
            )
            runtime.start(spec)
            mock_resolve.assert_called_once()

    def test_calls_build_process_spec(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=True)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=1.0,
            poll_interval=0.01,
        )
        with mock.patch.object(runtime, "_build_process_spec") as mock_build:
            mock_build.return_value = ProcessSpec(command=["test"])
            model = ModelSpec(local_path=Path("/tmp/model.pt"))
            Path("/tmp/model.pt").touch()
            spec = ServiceSpec(
                service_type="segmentation",
                model=model,
                port=8080,
                settings={"host": "127.0.0.1"},
            )
            runtime.start(spec)
            mock_build.assert_called_once()

    def test_calls_launch(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=True)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=1.0,
            poll_interval=0.01,
        )
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1"},
        )
        runtime.start(spec)
        assert len(launcher._processes) == 1

    def test_stops_on_early_exit(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=False)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=0.1,
            poll_interval=0.01,
        )
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1"},
        )
        with pytest.raises(SAMRuntimeError):
            runtime.start(spec)
        assert launcher._processes[0]._running is False

    def test_stops_on_timeout(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=False)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=0.1,
            poll_interval=0.01,
        )
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1"},
        )
        with pytest.raises(SAMRuntimeError):
            runtime.start(spec)
        assert launcher._processes[0]._running is False

    def test_stops_on_probe_exception(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=False)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=0.1,
            poll_interval=0.01,
        )
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1"},
        )
        with pytest.raises(SAMRuntimeError):
            runtime.start(spec)
        assert launcher._processes[0]._running is False


class TestStop:
    def test_delegates_to_launcher(self):
        launcher = FakeProcessLauncher()
        probe = FakeReadinessProbe(return_value=True)
        runtime = SAMRuntime(
            process_launcher=launcher,
            readiness_probe=probe,
            startup_timeout=1.0,
            poll_interval=0.01,
        )
        model = ModelSpec(local_path=Path("/tmp/model.pt"))
        Path("/tmp/model.pt").touch()
        spec = ServiceSpec(
            service_type="segmentation",
            model=model,
            port=8080,
            settings={"host": "127.0.0.1"},
        )
        service = runtime.start(spec)
        runtime.stop(service)
        assert launcher._processes[0]._running is False

    def test_raises_on_invalid_handle(self):
        launcher = FakeProcessLauncher()
        runtime = SAMRuntime(process_launcher=launcher)
        service = RunningService(
            spec=ServiceSpec(service_type="segmentation", model=None, port=8080, settings={}),
            endpoint=ServiceEndpoint(host="127.0.0.1", port=8080, service_type="segmentation"),
            runtime_handle="not a RunningProcess",
        )
        with pytest.raises(SAMRuntimeError, match="Invalid runtime handle"):
            runtime.stop(service)