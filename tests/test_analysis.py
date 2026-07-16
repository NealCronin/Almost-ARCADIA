from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from core.analysis import AnalysisCoordinator
from core.config import AppConfig, ConfigStore
from core.errors import AnalysisError, InferenceError
from core.pipeline.priority_map_adapter import PipelineResult
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

    def start(self, spec):
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
