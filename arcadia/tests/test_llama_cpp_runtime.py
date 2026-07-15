"""Tests for the llama-cpp-python LLM runtime."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from arcadia.contracts import ModelSpec, RunningService, ServiceEndpoint, ServiceSpec
from arcadia.process import ProcessLauncher, ProcessSpec, RunningProcess
from arcadia.runtimes.llama_cpp import LLMRuntime, LLMRuntimeError


# ── Fakes ──────────────────────────────────────────────────────────────────────

class FakeProcess:
    """A fake process that records its command and can be stopped."""

    def __init__(self, command: list[str]) -> None:
        self.command = command
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    @property
    def stdout_lines(self) -> list[str]:
        return []

    @property
    def stderr_lines(self) -> list[str]:
        return []


class FakeProcessLauncher:
    """A fake ProcessLauncher that returns a FakeProcess."""

    def __init__(self) -> None:
        self.stopped_processes: list[RunningProcess] = []

    def start(self, spec: ProcessSpec) -> RunningProcess:
        return RunningProcess(process=FakeProcess(spec.command), spec=spec)

    def stop(self, running: RunningProcess) -> None:
        self.stopped_processes.append(running)
        running.process.stop()


class FakeModelDownloader:
    """A fake model downloader that returns a path to a temp file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path("/tmp/fake_model.bin")
        self.call_count = 0

    def __call__(self, repo_id: str, filename: str) -> str:
        self.call_count += 1
        return str(self.path)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_spec(
    port: int = 8000,
    model: ModelSpec | None = None,
    settings: dict | None = None,
) -> ServiceSpec:
    return ServiceSpec(
        service_type="llm",
        port=port,
        model=model or ModelSpec(repository="test/model", filename="model.bin"),
        settings=settings or {},
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLLMRuntimeError:
    """LLMRuntimeError is a RuntimeError subclass."""

    def test_is_runtime_error(self) -> None:
        assert issubclass(LLMRuntimeError, RuntimeError)


class TestLLMRuntimeInit:
    """LLMRuntime stores constructor arguments."""

    def test_defaults(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        assert runtime._process_launcher is launcher
        assert runtime._python_executable == Path(__import__("sys").executable)
        assert runtime._startup_timeout == 120.0
        assert runtime._poll_interval == 0.25
        assert runtime._readiness_probe is runtime._default_readiness_probe

    def test_custom_python_executable(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher, python_executable="/usr/bin/my-python")

        assert runtime._python_executable == Path("/usr/bin/my-python")

    def test_custom_model_downloader(self) -> None:
        launcher = FakeProcessLauncher()
        downloader = FakeModelDownloader()
        runtime = LLMRuntime(process_launcher=launcher, model_downloader=downloader)

        assert runtime._model_downloader is downloader

    def test_custom_readiness_probe(self) -> None:
        launcher = FakeProcessLauncher()
        probe = lambda h, p: True
        runtime = LLMRuntime(process_launcher=launcher, readiness_probe=probe)

        assert runtime._readiness_probe is probe


class TestResolveModelPath:
    """_resolve_model_path returns the correct Path."""

    def test_local_path_exists(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model = ModelSpec(local_path="/tmp/existing.bin")
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(Path, "is_file", return_value=True):
                result = runtime._resolve_model_path(model)
                assert result == Path("/tmp/existing.bin")

    def test_local_path_not_found(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model = ModelSpec(local_path="/tmp/missing.bin")
        with mock.patch.object(Path, "exists", return_value=False):
            with mock.patch.object(Path, "is_file", return_value=False):
                with pytest.raises(LLMRuntimeError, match="not found"):
                    runtime._resolve_model_path(model)

    def test_repository_filename_downloads(self) -> None:
        launcher = FakeProcessLauncher()
        downloader = FakeModelDownloader(path=Path("/tmp/downloaded.bin"))
        runtime = LLMRuntime(process_launcher=launcher, model_downloader=downloader)

        model = ModelSpec(repository="test/model", filename="model.bin")
        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(Path, "is_file", return_value=True):
                result = runtime._resolve_model_path(model)
                assert result == Path("/tmp/downloaded.bin")
                assert downloader.call_count == 1

    def test_repository_filename_missing(self) -> None:
        launcher = FakeProcessLauncher()
        downloader = FakeModelDownloader(path=Path("/tmp/downloaded.bin"))
        runtime = LLMRuntime(process_launcher=launcher, model_downloader=downloader)

        model = ModelSpec(repository="test/model", filename="model.bin")
        with mock.patch.object(Path, "exists", return_value=False):
            with mock.patch.object(Path, "is_file", return_value=False):
                with pytest.raises(LLMRuntimeError, match="not found"):
                    runtime._resolve_model_path(model)

    def test_neither_local_path_nor_repository(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model = ModelSpec()
        with pytest.raises(LLMRuntimeError, match="must provide"):
            runtime._resolve_model_path(model)


class TestBuildProcessSpec:
    """_build_process_spec builds the correct command list."""

    def test_minimal(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000)

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert proc_spec.command == [
            str(runtime._python_executable),
            "-m",
            "llama_cpp.server",
            "--model",
            "/tmp/model.bin",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ]

    def test_host_from_settings(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"host": "0.0.0.0"})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--host" in proc_spec.command
        assert "0.0.0.0" in proc_spec.command

    def test_context_size(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"context_size": 4096})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--n_ctx" in proc_spec.command
        assert "4096" in proc_spec.command

    def test_batch_size(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"batch_size": 256})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--n_batch" in proc_spec.command
        assert "256" in proc_spec.command

    def test_microbatch_size(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"microbatch_size": 64})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--n_ubatch" in proc_spec.command
        assert "64" in proc_spec.command

    def test_gpu_layers(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"gpu_layers": 4})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--n_gpu_layers" in proc_spec.command
        assert "4" in proc_spec.command

    def test_threads(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"threads": 16})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--n_threads" in proc_spec.command
        assert "16" in proc_spec.command

    def test_threads_batch(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"threads_batch": 32})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--n_threads_batch" in proc_spec.command
        assert "32" in proc_spec.command

    def test_flash_attention_true(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"flash_attention": True})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--flash_attn" in proc_spec.command
        assert "true" in proc_spec.command

    def test_flash_attention_false(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"flash_attention": False})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--flash_attn" in proc_spec.command
        assert "false" in proc_spec.command

    def test_chat_format(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"chat_format": "llama2"})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--chat_format" in proc_spec.command
        assert "llama2" in proc_spec.command

    def test_model_projector(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        projector_path = Path("/tmp/clip.bin")
        spec = _make_spec(port=9000, settings={"model_projector": str(projector_path)})

        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(Path, "is_file", return_value=True):
                proc_spec = runtime._build_process_spec(spec, model_path)

                assert "--clip_model_path" in proc_spec.command
                assert str(projector_path) in proc_spec.command

    def test_model_projector_not_found(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"model_projector": "/tmp/missing.bin"})

        with mock.patch.object(Path, "exists", return_value=False):
            with mock.patch.object(Path, "is_file", return_value=False):
                with pytest.raises(LLMRuntimeError, match="not found"):
                    runtime._build_process_spec(spec, model_path)

    def test_extra_args(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"extra_args": ["--verbose", "--seed", "123"]})

        proc_spec = runtime._build_process_spec(spec, model_path)

        assert "--verbose" in proc_spec.command
        assert "--seed" in proc_spec.command
        assert "123" in proc_spec.command

    def test_extra_args_must_be_list(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"extra_args": "--verbose"})

        with pytest.raises(LLMRuntimeError, match="must be a list"):
            runtime._build_process_spec(spec, model_path)

    def test_extra_args_items_must_be_str(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        model_path = Path("/tmp/model.bin")
        spec = _make_spec(port=9000, settings={"extra_args": ["--verbose", 42]})

        with pytest.raises(LLMRuntimeError, match="must be a list of strings"):
            runtime._build_process_spec(spec, model_path)


class TestValidate:
    """_validate raises LLMRuntimeError for invalid specs."""

    def test_service_type_not_llm(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = ServiceSpec(
            service_type="web",
            port=8000,
            model=ModelSpec(repository="test/model", filename="model.bin"),
        )

        with pytest.raises(LLMRuntimeError, match="Expected service_type='llm'"):
            runtime._validate(spec)

    def test_model_is_none(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = ServiceSpec(
            service_type="llm",
            port=8000,
            model=None,
        )

        with pytest.raises(LLMRuntimeError, match="ModelSpec is required"):
            runtime._validate(spec)

    def test_port_out_of_range(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=0)
        with pytest.raises(LLMRuntimeError, match="Invalid port"):
            runtime._validate(spec)

        spec = _make_spec(port=70000)
        with pytest.raises(LLMRuntimeError, match="Invalid port"):
            runtime._validate(spec)

    def test_context_size_wrong_type(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"context_size": "4096"})

        with pytest.raises(LLMRuntimeError, match="must be int"):
            runtime._validate(spec)

    def test_batch_size_wrong_type(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"batch_size": 512.0})

        with pytest.raises(LLMRuntimeError, match="must be int"):
            runtime._validate(spec)

    def test_flash_attention_wrong_type(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"flash_attention": "true"})

        with pytest.raises(LLMRuntimeError, match="must be bool"):
            runtime._validate(spec)

    def test_host_empty_string(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"host": ""})

        with pytest.raises(LLMRuntimeError, match="must be non-empty"):
            runtime._validate(spec)

    def test_chat_format_empty_string(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"chat_format": ""})

        with pytest.raises(LLMRuntimeError, match="must be non-empty"):
            runtime._validate(spec)

    def test_model_projector_empty_string(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"model_projector": ""})

        with pytest.raises(LLMRuntimeError, match="must be non-empty"):
            runtime._validate(spec)

    def test_extra_args_not_list(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"extra_args": "--verbose"})

        with pytest.raises(LLMRuntimeError, match="must be a list"):
            runtime._validate(spec)

    def test_extra_args_items_not_str(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"extra_args": ["--verbose", 42]})

        with pytest.raises(LLMRuntimeError, match="must be a list of strings"):
            runtime._validate(spec)

    def test_unsupported_setting_parallel_slots(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"parallel_slots": 4})

        with pytest.raises(LLMRuntimeError, match="unsupported setting: parallel_slots"):
            runtime._validate(spec)

    def test_unsupported_setting_image_min_tokens(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"image_min_tokens": 256})

        with pytest.raises(LLMRuntimeError, match="unsupported setting: image_min_tokens"):
            runtime._validate(spec)

    def test_kv_cache_type_not_implemented(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"k_cache_type": 0})

        with pytest.raises(LLMRuntimeError, match="KV cache type"):
            runtime._validate(spec)

    def test_v_cache_type_not_implemented(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"v_cache_type": 0})

        with pytest.raises(LLMRuntimeError, match="KV cache type"):
            runtime._validate(spec)

    def test_unknown_setting(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"unknown_key": 1})

        with pytest.raises(LLMRuntimeError, match="unknown setting"):
            runtime._validate(spec)

    def test_multiple_unknown_settings(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={"a": 1, "b": 2})

        with pytest.raises(LLMRuntimeError, match="unknown setting"):
            runtime._validate(spec)

    def test_valid_spec_passes(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        spec = _make_spec(port=8000, settings={
            "context_size": 4096,
            "batch_size": 256,
            "microbatch_size": 64,
            "gpu_layers": 4,
            "threads": 16,
            "threads_batch": 32,
            "flash_attention": True,
            "host": "0.0.0.0",
            "chat_format": "llama2",
            "extra_args": ["--verbose"],
        })

        # Should not raise
        runtime._validate(spec)


class TestDefaultReadinessProbe:
    """_default_readiness_probe returns True when server responds."""

    def test_returns_true_on_success(self) -> None:
        with mock.patch("urllib.request.urlopen", return_value=mock.MagicMock()):
            result = LLMRuntime._default_readiness_probe("127.0.0.1", 8000)
            assert result is True

    def test_returns_false_on_connection_error(self) -> None:
        import urllib.error

        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = LLMRuntime._default_readiness_probe("127.0.0.1", 8000)
            assert result is False

    def test_returns_false_on_http_error(self) -> None:
        import urllib.error

        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError("127.0.0.1", 503, "Not Ready", {}, None)):
            result = LLMRuntime._default_readiness_probe("127.0.0.1", 8000)
            assert result is False

    def test_returns_false_on_os_error(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=OSError("Connection refused")):
            result = LLMRuntime._default_readiness_probe("127.0.0.1", 8000)
            assert result is False


class TestWaitForService:
    """_wait_for_service blocks until probe succeeds or timeout."""

    def test_returns_immediately_when_ready(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher, poll_interval=0.01)

        with mock.patch.object(runtime, "_readiness_probe", return_value=True):
            runtime._wait_for_service("127.0.0.1", 8000)
            # Should return immediately

    def test_raises_on_timeout(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher, startup_timeout=0.1, poll_interval=0.01)

        with mock.patch.object(runtime, "_readiness_probe", return_value=False):
            with pytest.raises(LLMRuntimeError, match="did not become ready"):
                runtime._wait_for_service("127.0.0.1", 8000)


class TestStart:
    """start() validates, resolves model, launches, waits, and returns RunningService."""

    def test_returns_running_service(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher, poll_interval=0.01)

        model_path = Path("/tmp/model.bin")
        with mock.patch.object(runtime, "_resolve_model_path", return_value=model_path):
            with mock.patch.object(runtime, "_wait_for_service"):
                spec = _make_spec(port=8000)
                service = runtime.start(spec)

                assert isinstance(service, RunningService)
                assert service.spec is spec
                assert service.endpoint.port == 8000
                assert service.endpoint.service_type == "llm"
                assert service.runtime_handle is not None

    def test_calls_validate(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        with mock.patch.object(runtime, "_validate") as mock_validate:
            with mock.patch.object(runtime, "_resolve_model_path"):
                with mock.patch.object(runtime, "_wait_for_service"):
                    runtime.start(_make_spec())

        mock_validate.assert_called_once()

    def test_calls_resolve_model_path(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        with mock.patch.object(runtime, "_validate"):
            with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")) as mock_resolve:
                with mock.patch.object(runtime, "_wait_for_service"):
                    runtime.start(_make_spec())

        mock_resolve.assert_called_once()

    def test_calls_build_process_spec(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        with mock.patch.object(runtime, "_validate"):
            with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
                with mock.patch.object(runtime, "_build_process_spec", return_value=ProcessSpec(command=[])) as mock_build:
                    with mock.patch.object(runtime, "_wait_for_service"):
                        runtime.start(_make_spec())

        mock_build.assert_called_once()

    def test_calls_process_launcher_start(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)
        with mock.patch.object(launcher, "start") as mock_start:
            with mock.patch.object(runtime, "_validate"):
                with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
                    with mock.patch.object(runtime, "_build_process_spec", return_value=ProcessSpec(command=[])):
                        with mock.patch.object(runtime, "_wait_for_service"):
                            runtime.start(_make_spec())
        mock_start.assert_called_once()

    def test_calls_wait_for_service(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        with mock.patch.object(runtime, "_validate"):
            with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
                with mock.patch.object(runtime, "_build_process_spec", return_value=ProcessSpec(command=[])):
                    with mock.patch.object(runtime, "_wait_for_service") as mock_wait:
                        runtime.start(_make_spec())

        mock_wait.assert_called_once()

    def test_passes_correct_host_and_port_to_wait(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        with mock.patch.object(runtime, "_validate"):
            with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
                with mock.patch.object(runtime, "_build_process_spec", return_value=ProcessSpec(command=[])):
                    with mock.patch.object(runtime, "_wait_for_service") as wait_mock:
                        runtime.start(_make_spec(port=9000, settings={"host": "0.0.0.0"}))

                    wait_mock.assert_called_once_with("0.0.0.0", 9000)


    """stop() calls ProcessLauncher.stop()."""

    def test_calls_stop(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        proc_spec = ProcessSpec(command=["python", "-m", "llama_cpp.server"])
        running_process = launcher.start(proc_spec)
        service = RunningService(
            spec=_make_spec(),
            endpoint=ServiceEndpoint(host="127.0.0.1", port=8000, service_type="llm"),
            runtime_handle=running_process,
        )

        runtime.stop(service)

        assert len(launcher.stopped_processes) == 1
        assert launcher.stopped_processes[0] is running_process
