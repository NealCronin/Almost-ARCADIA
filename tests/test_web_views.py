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


def llm_post_data(**overrides):
    data = {
        "node": "local",
        "inference_port": 8081,
        "bind_host": "0.0.0.0",
        "startup_timeout": 600,
        "model_source": "local",
        "model_path": "/models/model.gguf",
        "hf_repo": "",
        "hf_file": "",
        "hf_cache_dir": "",
        "n_ctx": 32768,
        "n_gpu_layers": -1,
        "n_threads": "",
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_parallel": 1,
        "flash_attn": "on",
        "cache_type_k": "",
        "cache_type_v": "",
        "chat_format": "",
        "model_alias": "local-model",
        "additional_arguments": "",
    }
    data.update(overrides)
    return data


def sam_post_data(**overrides):
    data = {
        "node": "local",
        "inference_port": 8090,
        "bind_host": "0.0.0.0",
        "startup_timeout": 600,
        "checkpoint": "/models/sam3.pt",
        "confidence": 0.25,
        "additional_arguments": "",
    }
    data.update(overrides)
    return data


def test_pages_load(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    assert client.get("/").status_code == 200
    assert client.get("/services/").status_code == 200
    assert client.get("/analysis/").status_code == 200
    assert client.get("/results/").status_code == 200


def test_service_forms_render_without_raw_json(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.get("/services/")
    assert response.status_code == 200
    assert b"settings_json" not in response.content
    assert b"Local model path" in response.content
    assert b"Checkpoint path" in response.content


def test_llm_service_post_delegates_and_persists(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/services/llm/start/", llm_post_data())
    assert response.status_code == 302
    assert runtime.controller.started[0].service_type == "llm"
    assert runtime.config_store.load().services["llm"].spec.settings["model_path"] == "/models/model.gguf"


def test_hugging_face_and_sam_posts_persist_builder_settings(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    hf_response = client.post(
        "/services/llm/start/",
        llm_post_data(model_source="huggingface", model_path="", hf_repo="org/model", hf_file="model.gguf"),
    )
    sam_response = client.post("/services/sam3/start/", sam_post_data())
    config = runtime.config_store.load()
    assert hf_response.status_code == 302
    assert sam_response.status_code == 302
    assert config.services["llm"].settings["hf_repo"] == "org/model"
    assert config.services["sam3"].settings["checkpoint"] == "/models/sam3.pt"


def test_invalid_service_form_returns_errors(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/services/llm/start/", llm_post_data(model_path=""))
    assert response.status_code == 400
    assert b"A local model path is required." in response.content


def test_service_post_is_blocked_during_analysis(client, monkeypatch, runtime):
    runtime.analysis.blocked = True
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/services/llm/start/", llm_post_data())
    assert response.status_code == 302
    assert runtime.controller.started == []

def test_analysis_post_returns_promptly(client, monkeypatch, runtime, tmp_path):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/analysis/start/", {"input_path": str(tmp_path)})
    assert response.status_code == 302
    assert runtime.analysis.started[0][0] == str(tmp_path)
