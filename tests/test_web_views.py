from __future__ import annotations

from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from core.config import AppConfig, ConfigStore, NodeConfig
from core.errors import AnalysisError
from core.services.specs import ServiceEndpoint
from web.runtime import ApplicationRuntime
from web.uploads import UploadStore


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
        self.active = False
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

    def is_active(self):
        return self.active


@pytest.fixture
def runtime(tmp_path: Path):
    config_path = tmp_path / "config.json"
    ConfigStore(config_path).save(AppConfig())
    value = ApplicationRuntime(
        ConfigStore(config_path), FakeController(), FakeAnalysis(), UploadStore(tmp_path / "uploads")
    )
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
    assert runtime.controller.started == []
    assert runtime.config_store.load().priority_map.services["llm"].spec.settings["model_path"] == "/models/model.gguf"


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
    assert config.priority_map.services["llm"].settings["hf_repo"] == "org/model"
    assert config.priority_map.services["sam3"].settings["checkpoint"] == "/models/sam3.pt"


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
    assert runtime.analysis.started[0][0] == tmp_path.resolve()


def test_host_create_edit_and_health_test(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    assert client.post("/host/nodes/", {"name": "worker-1"}).status_code == 302
    assert (
        client.post(
            "/host/nodes/worker-1/",
            {"mode": "remote", "host": "192.168.1.20", "instruction_port": 9001},
        ).status_code
        == 302
    )

    calls = []

    class HealthClient:
        def __init__(self, host, port, timeout, retries):
            calls.append((host, port, timeout, retries))

        def health(self):
            return True

    monkeypatch.setattr("web.views.InstructionClient", HealthClient)
    response = client.post("/host/nodes/worker-1/test/")
    assert response.json() == {"state": "reachable", "message": "Instruction server is reachable."}
    assert calls == [("192.168.1.20", 9001, 2.0, 0)]


def test_run_stream_and_artifacts_are_scoped_to_current_run(client, monkeypatch, runtime, tmp_path):
    from core.analysis import AnalysisStatus

    output_root = tmp_path / "outputs"
    run_id = "run-1"
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True)
    (run_directory / "preview.jpg").write_bytes(b"jpeg")
    (run_directory / "analysis.log").write_text("log", encoding="utf-8")
    config = runtime.config_store.load()
    config.priority_map.output.root = output_root
    runtime.config_store.save(config)

    class StreamAnalysis:
        def status(self):
            return AnalysisStatus(
                state="completed",
                run_id=run_id,
                stream_url=f"/client/priority-map/runs/{run_id}/stream/",
                artifacts_url=f"/client/priority-map/runs/{run_id}/artifacts/",
            )

        def preview(self, version):
            return b"jpeg", 1, "completed"

        def is_active(self):
            return False

    runtime.analysis = StreamAnalysis()
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    stream = client.get(f"/client/priority-map/runs/{run_id}/stream/")
    assert b"--frame\r\nContent-Type: image/jpeg\r\n" in b"".join(stream.streaming_content)
    artifacts = client.get(f"/client/priority-map/runs/{run_id}/artifacts/").json()
    assert artifacts["run_id"] == run_id
    assert artifacts["log"]["path"] == "analysis.log"
    assert client.get(f"/client/priority-map/runs/{run_id}/artifacts/preview.jpg/").status_code == 200
    download = client.get(f"/client/priority-map/runs/{run_id}/artifacts/preview.jpg/?download=1")
    assert download["Content-Disposition"].startswith("attachment;")


def test_upload_staging_requires_explicit_run_and_persists_manifest(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    upload = SimpleUploadedFile("mission.jpg", b"image", content_type="image/jpeg")

    staged = client.post(
        "/client/priority-map/uploads/",
        {"files": upload, "relative_paths": "mission.jpg"},
    )

    assert staged.status_code == 201
    payload = staged.json()["upload"]
    assert payload["source_type"] == "image"
    assert payload["file_count"] == 1
    assert payload["delete_url"] == f"/client/priority-map/uploads/{payload['id']}/delete/"
    assert runtime.analysis.started == []
    assert Client().get("/client/priority-map/uploads/").status_code == 200
    retained = Client().get("/client/priority-map/uploads/").json()["uploads"]
    assert [item["id"] for item in retained] == [payload["id"]]

    started = client.post("/client/priority-map/runs/", {"upload_id": payload["id"]})

    assert started.status_code == 302
    assert runtime.analysis.started[0][0] == runtime.uploads.input_path(payload["id"])


def test_upload_delete_is_explicit_safe_and_does_not_start_analysis(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    upload = SimpleUploadedFile("mission.jpg", b"image", content_type="image/jpeg")
    payload = client.post(
        "/client/priority-map/uploads/",
        {"files": upload, "relative_paths": "mission.jpg"},
    ).json()["upload"]
    input_path = runtime.uploads.input_path(payload["id"])

    assert client.get(payload["delete_url"]).status_code == 405
    assert client.post(payload["delete_url"]).status_code == 200
    assert not input_path.exists()
    assert runtime.analysis.started == []
    assert client.post("/client/priority-map/uploads/not-an-id/delete/").status_code == 400
    assert client.post("/client/priority-map/uploads/" + ("0" * 32) + "/delete/").status_code == 404


def test_active_upload_delete_is_rejected_without_removing_files(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    def stage(name: str) -> dict[str, object]:
        return client.post(
            "/client/priority-map/uploads/",
            {"files": SimpleUploadedFile(name, b"image", content_type="image/jpeg"), "relative_paths": name},
        ).json()["upload"]

    active, unrelated = stage("active.jpg"), stage("unrelated.jpg")
    active_path = runtime.uploads.input_path(str(active["id"]))
    runtime.analysis.active = True
    runtime.analysis._status.input_path = str(active_path)

    blocked = client.post(str(active["delete_url"]))
    unrelated_deleted = client.post(str(unrelated["delete_url"]))

    assert blocked.status_code == 409
    assert blocked.json() == {"detail": "Cannot delete the upload used by the active run."}
    assert active_path.exists()
    assert unrelated_deleted.status_code == 200
    runtime.analysis.active = False
    assert client.post(str(active["delete_url"])).status_code == 200


def test_host_form_errors_retain_all_host_cards_and_save_aliases(client, monkeypatch, runtime) -> None:
    config = runtime.config_store.load()
    config.nodes["worker-1"] = NodeConfig(mode="remote", host="192.168.1.20", instruction_port=9001)
    runtime.config_store.save(config)
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    assert client.get("/host/").status_code == 200
    assert client.get("/nodes/").status_code == 200
    assert (
        client.post("/host/nodes/local/", {"mode": "local", "host": "localhost", "instruction_port": ""}).status_code
        == 302
    )
    assert (
        client.post("/nodes/local/save/", {"mode": "local", "host": "localhost", "instruction_port": ""}).status_code
        == 302
    )

    responses = [
        (client.post("/host/nodes/", {"name": ""}), b"This field is required."),
        (client.post("/host/nodes/", {"name": "worker-1"}), b"A host with that name already exists."),
        (
            client.post(
                "/host/nodes/worker-1/",
                {"mode": "remote", "host": "192.168.1.20", "instruction_port": ""},
            ),
            b"Remote nodes require an instruction port.",
        ),
        (
            client.post(
                "/host/nodes/local/",
                {"mode": "remote", "host": "localhost", "instruction_port": "9000"},
            ),
            b"This computer must remain local.",
        ),
        (
            client.post(
                "/host/nodes/worker-1/",
                {"mode": "local", "host": "192.168.1.20", "instruction_port": ""},
            ),
            b"Configured hosts must remain remote.",
        ),
    ]
    for response, error in responses:
        assert response.status_code == 400
        assert error in response.content
        assert b"This computer" in response.content
        assert b"worker-1" in response.content
        assert b"Add host" in response.content


def test_results_explains_in_memory_runs_and_keeps_terminal_artifacts(client, monkeypatch, runtime, tmp_path) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    idle = client.get("/results/")
    assert idle.status_code == 200
    assert b"Current run" in idle.content
    assert b"live in-memory preview" in idle.content
    assert b"Artifacts remain available after the live preview ends." in idle.content

    from core.analysis import AnalysisStatus

    run_id = "cancelled-run"
    output_root = tmp_path / "outputs"
    (output_root / run_id).mkdir(parents=True)
    (output_root / run_id / "preview.jpg").write_bytes(b"jpeg")
    config = runtime.config_store.load()
    config.priority_map.output.root = output_root
    runtime.config_store.save(config)
    runtime.analysis._status = AnalysisStatus(
        state="cancelled",
        run_id=run_id,
        artifacts_url=f"/client/priority-map/runs/{run_id}/artifacts/",
    )

    artifacts = client.get(f"/client/priority-map/runs/{run_id}/artifacts/").json()

    assert artifacts["artifacts"][0]["inline_url"].endswith("/preview.jpg/")
    assert artifacts["artifacts"][0]["download_url"].endswith("/preview.jpg/?download=1")
