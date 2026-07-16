from __future__ import annotations

from pathlib import Path

import pytest

from core.config import AppConfig, ConfigStore
from core.errors import AnalysisError
from core.services.specs import ServiceEndpoint
from web.runtime import ApplicationRuntime


class FakeController:
    def __init__(self):
        self.started = []

    def list_services(self):
        return []

    def start(self, spec):
        self.started.append(spec)
        return ServiceEndpoint("127.0.0.1", spec.port, spec.service_type)

    def stop(self, port):
        return None

    def get_logs(self, port):
        return "log"


class FakeAnalysis:
    def __init__(self, blocked=False):
        self.blocked = blocked
        self.started = []
        from core.analysis import AnalysisStatus

        self._status = AnalysisStatus()

    def status(self):
        return self._status

    def assert_configuration_mutable(self):
        if self.blocked:
            raise AnalysisError("active")

    def start(self, path, config):
        self.started.append((path, config))


@pytest.fixture
def runtime(tmp_path: Path):
    config_path = tmp_path / "config.json"
    ConfigStore(config_path).save(AppConfig())
    value = ApplicationRuntime(ConfigStore(config_path), FakeController(), FakeAnalysis())
    return value


def test_pages_load(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    assert client.get("/").status_code == 200
    assert client.get("/services/").status_code == 200
    assert client.get("/analysis/").status_code == 200
    assert client.get("/results/").status_code == 200


def test_service_post_delegates_and_persists(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/services/llm/start/", {"node": "local", "inference_port": 8081, "settings_json": "{}"})
    assert response.status_code == 302
    assert runtime.controller.started[0].service_type == "llm"
    assert runtime.config_store.load().services["llm"].port == 8081


def test_service_post_is_blocked_during_analysis(client, monkeypatch, runtime):
    runtime.analysis.blocked = True
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/services/llm/start/", {"node": "local", "inference_port": 8081, "settings_json": "{}"})
    assert response.status_code == 302
    assert runtime.controller.started == []


def test_analysis_post_returns_promptly(client, monkeypatch, runtime, tmp_path):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/analysis/start/", {"input_path": str(tmp_path)})
    assert response.status_code == 302
    assert runtime.analysis.started[0][0] == str(tmp_path)
