"""Tests for the llama-cpp-python LLM runtime."""

from __future__ import annotations

import os
import sys
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

    def poll(self) -> int | None:
        return None if not self._stopped else 0


class FakeProcessLauncher:
    """A fake ProcessLauncher that returns a FakeProcess."""

    def __init__(self) -> None:
        self.stopped_processes: list[RunningProcess] = []
        self._alive: bool = True
        self._fake_stdout: list[str] = []
        self._fake_stderr: list[str] = []

    def start(self, spec: ProcessSpec) -> RunningProcess:
        return RunningProcess(process=FakeProcess(spec.command), spec=spec)

    def stop(self, running: RunningProcess) -> None:
        self.stopped_processes.append(running)
        running.process.stop()

    def is_running(self, running: RunningProcess) -> bool:
        return self._alive

    def recent_stdout(self, running: RunningProcess) -> list[str]:
        return list(self._fake_stdout)

    def recent_stderr(self, running: RunningProcess) -> list[str]:
        return list(self._fake_stderr)


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
    if model is None:
        model = ModelSpec(repository="test/model", filename="model.bin")
    return ServiceSpec(
        service_type="llm",
        port=port,
        model=model,
        settings=settings or {},
    )


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLLMRuntimeError:
    """LLMRuntimeError is a RuntimeError subclass."""

    def test_is_subclass_of_runtime_error(self) -> None:
        assert issubclass(LLMRuntimeError, RuntimeError)


class TestLLMRuntimeInit:
    """LLMRuntime stores constructor arguments."""

    def test_stores_process_launcher(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)
        assert runtime._process_launcher is launcher

    def test_stores_python_executable(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        assert runtime._python_executable == Path(sys.executable)
    def test_stores_custom_python_executable(self) -> None:
        runtime = LLMRuntime(
            process_launcher=FakeProcessLauncher(),
            python_executable=Path("/custom/python"),
        )
        assert runtime._python_executable == Path("/custom/python")

    def test_stores_model_downloader(self) -> None:
        downloader = FakeModelDownloader()
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher(), model_downloader=downloader)
        assert runtime._model_downloader is downloader

    def test_uses_default_model_downloader(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        assert runtime._model_downloader is LLMRuntime._default_model_downloader

    def test_stores_startup_timeout(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher(), startup_timeout=60.0)
        assert runtime._startup_timeout == 60.0

    def test_stores_poll_interval(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher(), poll_interval=0.5)
        assert runtime._poll_interval == 0.5

    def test_uses_default_readiness_probe(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        assert runtime._readiness_probe is LLMRuntime._default_readiness_probe

    def test_uses_custom_readiness_probe(self) -> None:
        probe = lambda h, p: True
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher(), readiness_probe=probe)
        assert runtime._readiness_probe is probe


class TestResolveModelPath:
    """_resolve_model_path returns the correct Path."""

    def test_returns_local_path(self) -> None:
        import tempfile
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        with tempfile.NamedTemporaryFile(suffix=".bin") as tf:
            model = ModelSpec(local_path=tf.name)
            assert runtime._resolve_model_path(model) == Path(tf.name)

    def test_raises_when_local_path_not_found(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model = ModelSpec(local_path="/nonexistent.bin")
        with pytest.raises(LLMRuntimeError, match="not found"):
            runtime._resolve_model_path(model)

    def test_downloads_from_hf(self) -> None:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bin") as tf:
            downloader = FakeModelDownloader(Path(tf.name))
            runtime = LLMRuntime(
                process_launcher=FakeProcessLauncher(),
                model_downloader=downloader,
            )
            model = ModelSpec(repository="test/model", filename="model.bin")
            assert runtime._resolve_model_path(model) == Path(tf.name)
            assert downloader.call_count == 1
    def test_raises_when_no_model_info(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model = ModelSpec()
        with pytest.raises(LLMRuntimeError, match="must provide"):
            runtime._resolve_model_path(model)


class TestBuildProcessSpec:
    """_build_process_spec builds the correct command list."""

    def test_basic_command(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec()
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert proc_spec.command[0] == sys.executable
        assert proc_spec.command[1] == "-m"
        assert proc_spec.command[2] == "llama_cpp.server"
        assert proc_spec.command[3] == "--model"
        assert proc_spec.command[4] == str(model_path)
        assert proc_spec.command[5] == "--host"
        assert proc_spec.command[6] == "127.0.0.1"
        assert proc_spec.command[7] == "--port"
        assert proc_spec.command[8] == "8000"

    def test_with_context_size(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"context_size": 4096})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--n_ctx" in proc_spec.command
        assert "4096" in proc_spec.command

    def test_with_batch_size(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"batch_size": 512})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--n_batch" in proc_spec.command
        assert "512" in proc_spec.command

    def test_with_microbatch_size(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"microbatch_size": 256})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--n_ubatch" in proc_spec.command
        assert "256" in proc_spec.command

    def test_with_gpu_layers(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"gpu_layers": 35})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--n_gpu_layers" in proc_spec.command
        assert "35" in proc_spec.command

    def test_with_threads(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"threads": 4})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--n_threads" in proc_spec.command
        assert "4" in proc_spec.command

    def test_with_threads_batch(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"threads_batch": 4})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--n_threads_batch" in proc_spec.command
        assert "4" in proc_spec.command

    def test_with_flash_attention_true(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"flash_attention": True})
        proc_spec = runtime._build_process_spec(spec, model_path)
        idx = proc_spec.command.index("--flash_attn")
        assert proc_spec.command[idx + 1] == "true"

    def test_with_flash_attention_false(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"flash_attention": False})
        proc_spec = runtime._build_process_spec(spec, model_path)
        idx = proc_spec.command.index("--flash_attn")
        assert proc_spec.command[idx + 1] == "false"

    def test_with_chat_format(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"chat_format": "chatml"})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--chat_format" in proc_spec.command
        assert "chatml" in proc_spec.command

    def test_uses_custom_host(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"host": "0.0.0.0"})
        proc_spec = runtime._build_process_spec(spec, model_path)
        idx = proc_spec.command.index("--host")
        assert proc_spec.command[idx + 1] == "0.0.0.0"

    def test_with_model_projector(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        proj_path = Path("/tmp/clip.pt")
        proj_path.write_text("dummy")
        try:
            spec = _make_spec(settings={"model_projector": str(proj_path)})
            proc_spec = runtime._build_process_spec(spec, model_path)
            assert "--clip_model_path" in proc_spec.command
            idx = proc_spec.command.index("--clip_model_path")
            assert proc_spec.command[idx + 1] == str(proj_path)
        finally:
            proj_path.unlink(missing_ok=True)

    def test_raises_when_projector_not_found(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"model_projector": "/nonexistent.pt"})
        with pytest.raises(LLMRuntimeError, match="not found"):
            runtime._build_process_spec(spec, model_path)

    def test_with_extra_args(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        extra = ["--no-mmap", "--mlock"]
        spec = _make_spec(settings={"extra_args": extra})
        proc_spec = runtime._build_process_spec(spec, model_path)
        for arg in extra:
            assert arg in proc_spec.command

    def test_raises_when_extra_args_not_list(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"extra_args": "not-a-list"})
        with pytest.raises(LLMRuntimeError, match="must be a list"):
            runtime._build_process_spec(spec, model_path)

    def test_raises_when_extra_args_contains_non_string(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"extra_args": [42]})
        with pytest.raises(LLMRuntimeError, match="must be a list"):
            runtime._build_process_spec(spec, model_path)

    def test_multiple_integer_settings(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"context_size": 2048, "gpu_layers": 20, "threads": 8})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--n_ctx" in proc_spec.command
        assert "2048" in proc_spec.command
        assert "--n_gpu_layers" in proc_spec.command
        assert "20" in proc_spec.command
        assert "--n_threads" in proc_spec.command
        assert "8" in proc_spec.command

    def test_does_not_mutate_spec_settings(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        settings = {"context_size": 4096}
        spec = _make_spec(settings=settings)
        runtime._build_process_spec(spec, model_path)
        assert settings == {"context_size": 4096}

    def test_does_not_include_unsupported_keys(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={"context_size": 4096, "parallel_slots": 4})
        proc_spec = runtime._build_process_spec(spec, model_path)
        assert "--parallel_slots" not in proc_spec.command

    def test_no_extra_args_when_not_provided(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        model_path = Path("/tmp/model.bin")
        spec = _make_spec(settings={})
        proc_spec = runtime._build_process_spec(spec, model_path)
        # Only the standard args should be present
        assert len(proc_spec.command) == 9


class TestValidate:
    """_validate raises LLMRuntimeError for invalid specs."""

    def test_raises_when_service_type_not_llm(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = ServiceSpec(service_type="embedding", port=8000, model=ModelSpec(repository="test/model", filename="model.bin"), settings={})
        with pytest.raises(LLMRuntimeError, match="Expected service_type='llm'"):
            runtime._validate(spec)

    def test_raises_when_model_is_none(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = ServiceSpec(service_type="llm", port=8000, model=None, settings={})
        with pytest.raises(LLMRuntimeError, match="ModelSpec is required"):
            runtime._validate(spec)

    def test_raises_when_port_out_of_range_low(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = ServiceSpec(service_type="llm", port=0, model=ModelSpec(repository="test/model", filename="model.bin"), settings={})
        with pytest.raises(LLMRuntimeError, match="Invalid port"):
            runtime._validate(spec)

    def test_raises_when_port_out_of_range_high(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = ServiceSpec(service_type="llm", port=65536, model=ModelSpec(repository="test/model", filename="model.bin"), settings={})
        with pytest.raises(LLMRuntimeError, match="Invalid port"):
            runtime._validate(spec)

    def test_raises_when_context_size_not_int(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"context_size": "4096"})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_batch_size_not_int(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"batch_size": "512"})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_microbatch_size_not_int(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"microbatch_size": "256"})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_gpu_layers_not_int(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"gpu_layers": "35"})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_threads_not_int(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"threads": "4"})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_threads_batch_not_int(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"threads_batch": "4"})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_flash_attention_not_bool(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"flash_attention": "true"})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_host_not_str(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"host": 123})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_host_empty(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"host": ""})
        with pytest.raises(LLMRuntimeError, match="must be non-empty"):
            runtime._validate(spec)

    def test_raises_when_chat_format_not_str(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"chat_format": 42})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_chat_format_empty(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"chat_format": ""})
        with pytest.raises(LLMRuntimeError, match="must be non-empty"):
            runtime._validate(spec)

    def test_raises_when_model_projector_not_str(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"model_projector": 42})
        with pytest.raises(LLMRuntimeError, match="must be"):
            runtime._validate(spec)

    def test_raises_when_model_projector_empty(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"model_projector": ""})
        with pytest.raises(LLMRuntimeError, match="must be non-empty"):
            runtime._validate(spec)

    def test_raises_when_extra_args_not_list(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"extra_args": "not-a-list"})
        with pytest.raises(LLMRuntimeError, match="must be a list"):
            runtime._validate(spec)

    def test_raises_when_extra_args_contains_non_string(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"extra_args": [42]})
        with pytest.raises(LLMRuntimeError, match="must be a list"):
            runtime._validate(spec)

    def test_raises_for_unsupported_parallel_slots(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"parallel_slots": 4})
        with pytest.raises(LLMRuntimeError, match="unsupported setting"):
            runtime._validate(spec)

    def test_raises_for_unsupported_image_min_tokens(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"image_min_tokens": 50})
        with pytest.raises(LLMRuntimeError, match="unsupported setting"):
            runtime._validate(spec)

    def test_raises_for_k_cache_type(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"k_cache_type": "f16"})
        with pytest.raises(LLMRuntimeError, match="not implemented"):
            runtime._validate(spec)

    def test_raises_for_v_cache_type(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"v_cache_type": "f16"})
        with pytest.raises(LLMRuntimeError, match="not implemented"):
            runtime._validate(spec)

    def test_raises_for_unknown_setting(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={"unknown_key": "value"})
        with pytest.raises(LLMRuntimeError, match="unknown setting"):
            runtime._validate(spec)

    def test_passes_with_valid_settings(self) -> None:
        runtime = LLMRuntime(process_launcher=FakeProcessLauncher())
        spec = _make_spec(settings={
            "context_size": 4096,
            "batch_size": 512,
            "flash_attention": True,
            "host": "0.0.0.0",
        })
        runtime._validate(spec)  # Should not raise


class TestDefaultReadinessProbe:
    """_default_readiness_probe returns True when server responds."""

    def test_returns_true_when_server_responds(self) -> None:
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock.MagicMock()
            assert LLMRuntime._default_readiness_probe("127.0.0.1", 8000) is True
            mock_urlopen.assert_called_once_with("http://127.0.0.1:8000/v1/models", timeout=2)

    def test_returns_false_on_connection_error(self) -> None:
        from urllib.error import URLError
        with mock.patch("urllib.request.urlopen", side_effect=URLError("connection refused")):
            assert LLMRuntime._default_readiness_probe("127.0.0.1", 8000) is False

    def test_returns_false_on_http_error(self) -> None:
        from urllib.error import HTTPError
        with mock.patch("urllib.request.urlopen", side_effect=HTTPError("http://example.com", 500, "Internal Server Error", {}, None)):
            assert LLMRuntime._default_readiness_probe("127.0.0.1", 8000) is False

    def test_returns_false_on_os_error(self) -> None:
        with mock.patch("urllib.request.urlopen", side_effect=OSError("connection reset")):
            assert LLMRuntime._default_readiness_probe("127.0.0.1", 8000) is False

    def test_uses_correct_url(self) -> None:
        with mock.patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = mock.MagicMock()
            LLMRuntime._default_readiness_probe("0.0.0.0", 9000)
            mock_urlopen.assert_called_once_with("http://0.0.0.0:9000/v1/models", timeout=2)


class TestWaitForService:
    """_wait_for_service blocks until probe succeeds or timeout."""

    def test_returns_immediately_when_ready(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher, poll_interval=0.01)
        running_process = launcher.start(ProcessSpec(command=["python"]))

        with mock.patch.object(runtime, "_readiness_probe", return_value=True):
            runtime._wait_for_service(running_process, "127.0.0.1", 8000)
            # Should return immediately

    def test_raises_on_timeout(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher, startup_timeout=0.1, poll_interval=0.01)
        running_process = launcher.start(ProcessSpec(command=["python"]))

        with mock.patch.object(runtime, "_readiness_probe", return_value=False):
            with pytest.raises(LLMRuntimeError, match="did not become ready"):
                runtime._wait_for_service(running_process, "127.0.0.1", 8000)


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

    def test_passes_host_and_port_to_wait(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)

        with mock.patch.object(runtime, "_validate"):
            with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
                with mock.patch.object(runtime, "_build_process_spec", return_value=ProcessSpec(command=[])):
                    with mock.patch.object(runtime, "_wait_for_service") as wait_mock:
                        runtime.start(_make_spec(port=9000, settings={"host": "0.0.0.0"}))

                    wait_mock.assert_called_once_with(mock.ANY, "0.0.0.0", 9000)

    def test_start_successful_readiness(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher, poll_interval=0.01)
        probe_results = iter([False, False, True])
        with mock.patch.object(runtime, "_readiness_probe", side_effect=lambda h, p: next(probe_results)):
            with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
                service = runtime.start(_make_spec(port=8000))
        assert isinstance(service, RunningService)
        assert len(launcher.stopped_processes) == 0
        assert isinstance(service.runtime_handle, RunningProcess)

    def test_start_early_exit(self) -> None:
        launcher = FakeProcessLauncher()
        launcher._alive = False
        launcher._fake_stderr = ["error: failed to load model"]
        runtime = LLMRuntime(process_launcher=launcher, poll_interval=0.01)
        with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
            with pytest.raises(LLMRuntimeError, match="failed to load model|exited before becoming ready"):
                runtime.start(_make_spec(port=8000))
        assert len(launcher.stopped_processes) == 1

    def test_start_timeout_cleanup(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher, startup_timeout=0.1, poll_interval=0.01)
        with mock.patch.object(runtime, "_readiness_probe", return_value=False):
            with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
                with pytest.raises(LLMRuntimeError, match="did not become ready"):
                    runtime.start(_make_spec(port=8000))
        assert len(launcher.stopped_processes) == 1

    def test_start_probe_exception_cleanup(self) -> None:
        launcher = FakeProcessLauncher()
        probe_error = RuntimeError("probe crashed")
        runtime = LLMRuntime(process_launcher=launcher, poll_interval=0.01)
        with mock.patch.object(runtime, "_readiness_probe", side_effect=probe_error):
            with mock.patch.object(runtime, "_resolve_model_path", return_value=Path("/tmp/m.bin")):
                with pytest.raises(LLMRuntimeError, match="Failed while waiting") as exc_info:
                    runtime.start(_make_spec(port=8000))
        assert exc_info.value.__cause__ is probe_error
        assert len(launcher.stopped_processes) == 1

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

    def test_stop_invalid_handle_none(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)
        service = RunningService(
            spec=_make_spec(),
            endpoint=ServiceEndpoint(host="127.0.0.1", port=8000, service_type="llm"),
            runtime_handle=None,
        )
        with pytest.raises(LLMRuntimeError, match="must be a RunningProcess"):
            runtime.stop(service)

    def test_stop_invalid_handle_object(self) -> None:
        launcher = FakeProcessLauncher()
        runtime = LLMRuntime(process_launcher=launcher)
        service = RunningService(
            spec=_make_spec(),
            endpoint=ServiceEndpoint(host="127.0.0.1", port=8000, service_type="llm"),
            runtime_handle=object(),
        )
        with pytest.raises(LLMRuntimeError, match="must be a RunningProcess"):
            runtime.stop(service)