from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.analysis import AnalysisCoordinator
from core.config import AppConfig, ConfigStore
from core.errors import AnalysisError, InferenceError, ServiceError, ServiceStartupError
from core.pipeline.priority_map_adapter import PipelineResult, PriorityMapAdapter
from core.services.specs import ServiceEndpoint


def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig.from_dict(
        {
            "nodes": {"local": {"mode": "local", "host": "127.0.0.1"}},
            "services": {
                "llm": {"node": "local", "service_type": "llm", "port": 8081, "settings": {"command": ["fake"]}},
                "sam3": {"node": "local", "service_type": "sam3", "port": 8090, "settings": {"command": ["fake"]}},
            },
            "output_root": str(tmp_path / "outputs"),
        }
    )


class FakeController:
    def __init__(self):
        self.started = []

    def is_running(self, port):
        return False

    def start(self, spec, *, cancel_event=None):
        self.started.append(spec)
        return ServiceEndpoint("127.0.0.1", spec.port, spec.service_type)


class FakeAdapter:
    def __init__(self):
        self.calls = 0
        self.release = None

    def run(self, **kwargs):
        self.calls += 1
        if self.release is not None:
            self.release.wait(timeout=5)
        kwargs["progress_callback"]({"frames_processed": 1, "frame_index": 0})
        return PipelineResult(
            kwargs["output_directory"], {"ok": True}, [kwargs["output_directory"] + "/partial.txt"], 1
        )


def test_analysis_writes_effective_settings_log_and_output(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    adapter = FakeAdapter()
    coordinator = AnalysisCoordinator(ConfigStore(tmp_path / "config.json"), FakeController(), adapter)
    status = coordinator.run_sync(source, app_config(tmp_path))
    assert status.state == "completed"
    output = Path(status.output_directory)
    assert (output / "analysis.log").exists()
    assert (output / "effective_settings.json").exists()
    assert status.frames_processed == 1


def test_analysis_rejects_second_active_run(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    adapter = FakeAdapter()
    adapter.release = threading.Event()
    coordinator = AnalysisCoordinator(None, FakeController(), adapter)
    coordinator.start(source, app_config(tmp_path))
    deadline = time.time() + 2
    while not coordinator.is_active() and time.time() < deadline:
        time.sleep(0.01)
    with pytest.raises(AnalysisError):
        coordinator.start(source, app_config(tmp_path))
    adapter.release.set()
    deadline = time.time() + 2
    while coordinator.is_active() and time.time() < deadline:
        time.sleep(0.01)
    assert coordinator.status().state == "completed"


def test_analysis_restarts_once_after_inference_failure(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()

    class RetryAdapter(FakeAdapter):
        def run(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise InferenceError("connection lost")
            return PipelineResult(kwargs["output_directory"], {}, [], 0)

    controller = FakeController()
    adapter = RetryAdapter()
    coordinator = AnalysisCoordinator(None, controller, adapter)
    status = coordinator.run_sync(source, app_config(tmp_path))
    assert status.state == "completed"
    assert adapter.calls == 2
    assert len(controller.started) == 4


@pytest.mark.parametrize(
    ("service_type", "expected_ports"),
    [
        ("llm", [8081, 8090, 8081]),
        ("sam3", [8081, 8090, 8090]),
        (None, [8081, 8090, 8081, 8090]),
    ],
)
def test_analysis_restarts_only_attributed_service(
    tmp_path: Path,
    service_type: str | None,
    expected_ports: list[int],
) -> None:
    source = tmp_path / "images"
    source.mkdir()

    class RetryAdapter(FakeAdapter):
        def run(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise InferenceError("connection lost", service_type=service_type)
            return PipelineResult(kwargs["output_directory"], {}, [], 0)

    controller = FakeController()
    adapter = RetryAdapter()
    status = AnalysisCoordinator(None, controller, adapter).run_sync(source, app_config(tmp_path))

    assert status.state == "completed"
    assert adapter.calls == 2
    assert [spec.port for spec in controller.started] == expected_ports


def test_analysis_stops_after_one_retry(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()

    class AlwaysFailAdapter(FakeAdapter):
        def run(self, **kwargs):
            self.calls += 1
            raise InferenceError("connection lost", service_type="llm")

    adapter = AlwaysFailAdapter()
    status = AnalysisCoordinator(None, FakeController(), adapter).run_sync(source, app_config(tmp_path))

    assert status.state == "failed"
    assert adapter.calls == 2


def test_rapid_sequential_analyses_use_distinct_output_directories(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    coordinator = AnalysisCoordinator(None, FakeController(), FakeAdapter())

    first = coordinator.run_sync(source, app_config(tmp_path))
    second = coordinator.run_sync(source, app_config(tmp_path))

    assert first.output_directory != second.output_directory
    assert Path(first.output_directory).is_dir()
    assert Path(second.output_directory).is_dir()


def test_cancel_processes_current_frame_preserves_partial_output_and_closes_runner(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    (source / "frame.jpg").write_bytes(b"jpeg")
    started = threading.Event()
    release = threading.Event()

    class FrameRunner:
        frames_processed = 0

        def has_next(self):
            return self.frames_processed < 2

        def run_frame(self):
            self.frames_processed += 1
            started.set()
            release.wait(timeout=2)
            return SimpleNamespace(frame_index=0, image_name="frame.jpg", keep_running=True)

        def result(self):
            return SimpleNamespace(frames_processed=self.frames_processed)

        def close(self):
            self.closed = True

    runner = FrameRunner()
    adapter = PriorityMapAdapter(runner_factory=lambda **_: runner)
    coordinator = AnalysisCoordinator(None, FakeController(), adapter)
    coordinator.start(source, app_config(tmp_path))
    assert started.wait(timeout=2)
    cancelling = coordinator.cancel_after_current_frame()
    assert cancelling.state == "cancelling"
    release.set()
    deadline = time.monotonic() + 2
    while coordinator.is_active() and time.monotonic() < deadline:
        time.sleep(0.01)

    status = coordinator.status()
    assert status.state == "cancelled"
    assert status.frames_processed == 1
    assert runner.closed
    assert status.stream_url == f"/client/priority-map/runs/{status.run_id}/stream/"
    assert status.artifacts_url == f"/client/priority-map/runs/{status.run_id}/artifacts/"
    assert Path(status.output_directory, "analysis.log").exists()


def test_cancel_while_llm_startup_waits_starts_no_sam_or_adapter(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    started = threading.Event()

    class BlockingController(FakeController):
        def start(self, spec, *, cancel_event=None):
            self.started.append(spec)
            if spec.service_type == "llm":
                started.set()
                assert cancel_event is not None
                cancel_event.wait(timeout=2)
                raise ServiceStartupError("LLM startup cancelled.")
            return ServiceEndpoint("127.0.0.1", spec.port, spec.service_type)

    controller = BlockingController()
    adapter = FakeAdapter()
    coordinator = AnalysisCoordinator(None, controller, adapter)
    coordinator.start(source, app_config(tmp_path))
    assert started.wait(timeout=2)
    cancelling = coordinator.cancel_after_current_frame()
    assert cancelling.message == "Cancelling after current startup step"

    deadline = time.monotonic() + 2
    while coordinator.is_active() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert coordinator.status().state == "cancelled"
    assert [spec.service_type for spec in controller.started] == ["llm"]
    assert adapter.calls == 0


def test_cancel_between_llm_and_sam_starts_no_sam(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    adapter = FakeAdapter()
    coordinator: AnalysisCoordinator

    class CancellingController(FakeController):
        def start(self, spec, *, cancel_event=None):
            self.started.append(spec)
            if spec.service_type == "llm":
                coordinator.cancel_after_current_frame()
            return ServiceEndpoint("127.0.0.1", spec.port, spec.service_type)

    controller = CancellingController()
    coordinator = AnalysisCoordinator(None, controller, adapter)
    status = coordinator.run_sync(source, app_config(tmp_path))

    assert status.state == "cancelled"
    assert [spec.service_type for spec in controller.started] == ["llm"]
    assert adapter.calls == 0


def test_cancel_while_sam_startup_waits_starts_no_adapter(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    started = threading.Event()

    class BlockingController(FakeController):
        def start(self, spec, *, cancel_event=None):
            self.started.append(spec)
            if spec.service_type == "sam3":
                started.set()
                assert cancel_event is not None
                cancel_event.wait(timeout=2)
                raise ServiceStartupError("SAM3 startup cancelled.")
            return ServiceEndpoint("127.0.0.1", spec.port, spec.service_type)

    controller = BlockingController()
    adapter = FakeAdapter()
    coordinator = AnalysisCoordinator(None, controller, adapter)
    coordinator.start(source, app_config(tmp_path))
    assert started.wait(timeout=2)
    coordinator.cancel_after_current_frame()

    deadline = time.monotonic() + 2
    while coordinator.is_active() and time.monotonic() < deadline:
        time.sleep(0.01)

    assert coordinator.status().state == "cancelled"
    assert [spec.service_type for spec in controller.started] == ["llm", "sam3"]
    assert adapter.calls == 0


def test_remote_readiness_wait_stops_after_cancelled_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    endpoint = ServiceEndpoint("127.0.0.1", 8081, "llm")
    cancelled = threading.Event()
    probes = 0

    def probe(*_, **__):
        nonlocal probes
        probes += 1
        cancelled.set()
        return SimpleNamespace(status_code=503)

    monkeypatch.setattr("core.analysis.requests.get", probe)

    with pytest.raises(ServiceError, match="startup cancelled"):
        AnalysisCoordinator._wait_ready(endpoint, 30, cancel_event=cancelled)

    assert probes == 1
