from __future__ import annotations

from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from core.config import AppConfig, ConfigStore, NodeConfig
from core.errors import AnalysisError
from core.services.host_listener import HostListenerRestartError, HostListenerStatus
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


class FakeHostListener:
    def __init__(self) -> None:
        self.restarts = []
        self.error: HostListenerRestartError | None = None
        self._status = HostListenerStatus(state="running", pid=123, message="Instruction server is running")

    def status(self) -> HostListenerStatus:
        return self._status

    def restart(self, config, *, rollback_config=None) -> HostListenerStatus:
        self.restarts.append((config, rollback_config))
        if self.error is not None:
            raise self.error
        self._status = HostListenerStatus(
            state="running",
            host=config.host,
            port=config.port,
            pid=456,
            message="Instruction server is running",
        )
        return self._status

    def close(self) -> None:
        return None


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
    listener = FakeHostListener()
    value = ApplicationRuntime(
        ConfigStore(config_path), listener, FakeController(), FakeAnalysis(), UploadStore(tmp_path / "uploads")
    )
    return value


def llm_post_data(**overrides):
    data = {
        "node": "local",
        "inference_port": 8081,
        "hf_repo": "org/model",
        "n_ctx": 32768,
        "temperature": 0.2,
        "top_k": 40,
        "min_p": 0.05,
        "top_p": 0.95,
        "local_bind_host": "127.0.0.1",
        "startup_timeout": 600,
        "n_gpu_layers": -1,
        "n_batch": 2048,
        "n_ubatch": 512,
        "flash_attn": "on",
        "offload_kqv": "on",
        "use_mmap": "on",
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


def remote_health(monkeypatch, reachable: bool) -> list[tuple[str, int, float, int]]:
    calls: list[tuple[str, int, float, int]] = []

    class FakeInstructionClient:
        def __init__(self, host, port, *, timeout, retries):
            calls.append((host, port, timeout, retries))

        def health(self):
            return reachable

    monkeypatch.setattr("web.views.InstructionClient", FakeInstructionClient)
    return calls


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
    assert b"Hugging Face repository" in response.content
    assert b"Checkpoint path" in response.content


def test_llm_service_post_delegates_and_persists(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/services/llm/start/", llm_post_data())
    assert response.status_code == 302
    assert runtime.controller.started == []
    assert runtime.config_store.load().priority_map.services["llm"].spec.settings["hf_repo"] == "org/model"


def test_hugging_face_and_sam_posts_persist_builder_settings(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    hf_response = client.post(
        "/services/llm/start/", llm_post_data(hf_repo="org/model", model_file_pattern="model.gguf")
    )
    sam_response = client.post("/services/sam3/start/", sam_post_data())
    config = runtime.config_store.load()
    assert hf_response.status_code == 302
    assert sam_response.status_code == 302
    assert config.priority_map.services["llm"].settings["hf_repo"] == "org/model"
    assert config.priority_map.services["sam3"].settings["checkpoint"] == "/models/sam3.pt"


def test_invalid_service_form_returns_errors(client, monkeypatch, runtime):
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    response = client.post("/services/llm/start/", llm_post_data(hf_repo=""))
    assert response.status_code == 400
    assert b"This field is required." in response.content
    assert b"Compute nodes" in response.content
    assert b"This computer" in response.content


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


def test_host_page_configures_only_this_instruction_server(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    response = client.get("/host/")

    assert response.status_code == 200
    assert b"IP address" in response.content
    assert b"Instruction port" in response.content
    assert b"Listening on 127.0.0.1:9000" in response.content
    assert b"Add remote host" not in response.content
    assert b"Compute hosts" not in response.content
    assert b"data-host-listener-status-url" in response.content
    assert b"data-host-listener-save" in response.content
    assert client.post("/host/nodes/", {"name": "worker-1"}).status_code == 404


def test_host_save_restarts_then_persists_listener_config(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    response = client.post("/host/save/", {"host": "127.0.0.1", "port": 9010})

    assert response.status_code == 302
    config = runtime.config_store.load()
    assert config.host_listener.host == "127.0.0.1"
    assert config.host_listener.port == 9010
    replacement, previous = runtime.host_listener.restarts[0]
    assert replacement.port == 9010
    assert previous.port == 9000
    host_page = client.get("/host/")
    assert b"Compute nodes" not in host_page.content
    assert b"Add remote computer" not in host_page.content


def test_host_save_preserves_unknown_listener_configuration(client, monkeypatch, runtime) -> None:
    config = runtime.config_store.load()
    config.host_listener.extra = {"future_listener_option": {"keep": True}}
    runtime.config_store.save(config)
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    response = client.post("/host/save/", {"host": "127.0.0.1", "port": 9010})

    assert response.status_code == 302
    assert runtime.config_store.load().host_listener.extra == {"future_listener_option": {"keep": True}}


def test_host_save_rolls_back_listener_when_configuration_persistence_fails(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    def fail_save(_config) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(runtime.config_store, "save", fail_save)
    response = client.post("/host/save/", {"host": "127.0.0.1", "port": 9010})

    assert response.status_code == 500
    assert b"The previous listener was restored." in response.content
    assert runtime.config_store.load().host_listener.port == 9000
    assert [replacement.port for replacement, _previous in runtime.host_listener.restarts] == [9010, 9000]


def test_landing_keeps_client_before_host_without_navigation(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    response = client.get("/")

    assert response.status_code == 200
    assert response.content.index(b"Client") < response.content.index(b"Host")
    assert b"<nav" not in response.content


def test_host_failed_replacement_keeps_saved_config_and_reports_rollback(client, monkeypatch, runtime) -> None:
    runtime.host_listener.error = HostListenerRestartError(
        "Replacement failed: unavailable. Previous instruction server was restored.", rollback_succeeded=True
    )
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    response = client.post("/host/save/", {"host": "127.0.0.1", "port": 9010})

    assert response.status_code == 502
    assert b"Previous instruction server was restored." in response.content
    assert runtime.config_store.load().host_listener.port == 9000


def test_host_listener_status_json_and_csrf_protection(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    status = client.get("/host/status/")

    assert status.json() == {
        "state": "running",
        "host": "127.0.0.1",
        "port": 9000,
        "pid": 123,
        "uptime_seconds": None,
        "message": "Instruction server is running",
        "health_url": "http://127.0.0.1:9000/health",
        "last_error": None,
    }
    csrf_client = Client(enforce_csrf_checks=True)
    assert csrf_client.post("/host/save/", {"host": "127.0.0.1", "port": 9010}).status_code == 403


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


def test_models_page_renders_compute_nodes_and_model_forms(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    response = client.get("/client/priority-map/models/")

    assert response.status_code == 200
    assert b"Compute nodes" in response.content
    assert b"This computer" in response.content
    assert b"Add remote computer" in response.content
    assert b"LLM service" in response.content
    assert b"SAM3 service" in response.content


def test_add_reachable_remote_node_preserves_config_and_updates_model_choices(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    calls = remote_health(monkeypatch, reachable=True)
    config = runtime.config_store.load()
    config.host_listener.extra = {"listener_extension": True}
    config.extra = {"root_extension": {"preserve": True}}
    runtime.config_store.save(config)

    response = client.post(
        "/client/priority-map/models/nodes/add/",
        {"name": "GPU Desktop", "host": "192.168.1.20", "instruction_port": 9000},
    )
    saved = runtime.config_store.load()
    page = client.get("/client/priority-map/models/")

    assert response.status_code == 302
    assert saved.nodes["gpu-desktop"] == NodeConfig("remote", "192.168.1.20", 9000)
    assert saved.host_listener.extra == {"listener_extension": True}
    assert saved.extra == {"root_extension": {"preserve": True}}
    assert calls == [("192.168.1.20", 9000, 2.0, 0)]
    assert page.content.count(b'value="gpu-desktop"') == 2


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"name": "local", "host": "192.168.1.20", "instruction_port": 9000}, b"reserved"),
        ({"name": "desktop", "host": "not-an-ip", "instruction_port": 9000}, b"valid IPv4"),
        ({"name": "desktop", "host": "192.168.1.20", "instruction_port": 0}, b"greater than or equal to 1"),
    ],
)
def test_add_remote_node_validation_keeps_complete_models_context(client, monkeypatch, runtime, data, message) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)

    response = client.post("/client/priority-map/models/nodes/add/", data)

    assert response.status_code == 400
    assert message in response.content
    assert b"Compute nodes" in response.content
    assert b"LLM service" in response.content
    assert b"SAM3 service" in response.content


def test_duplicate_remote_node_name_is_rejected(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    remote_health(monkeypatch, reachable=True)
    config = runtime.config_store.load()
    config.nodes["desktop"] = NodeConfig("remote", "192.168.1.20", 9000)
    runtime.config_store.save(config)

    response = client.post(
        "/client/priority-map/models/nodes/add/",
        {"name": "desktop", "host": "192.168.1.21", "instruction_port": 9001},
    )

    assert response.status_code == 400
    assert b"already exists" in response.content
    assert runtime.config_store.load().nodes["desktop"].host == "192.168.1.20"


def test_unreachable_remote_node_requires_explicit_save_anyway(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    remote_health(monkeypatch, reachable=False)
    payload = {"name": "desktop", "host": "192.168.1.20", "instruction_port": 9000}

    first = client.post("/client/priority-map/models/nodes/add/", payload)

    assert first.status_code == 400
    assert b"Instruction server is unreachable" in first.content
    assert b"Save anyway" in first.content
    assert "desktop" not in runtime.config_store.load().nodes
    saved = client.post("/client/priority-map/models/nodes/add/", {**payload, "save_anyway": "1"})
    assert saved.status_code == 302
    assert runtime.config_store.load().nodes["desktop"].host == "192.168.1.20"


def test_remote_node_save_failure_reports_error_without_persisting(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    remote_health(monkeypatch, reachable=True)
    monkeypatch.setattr(runtime.config_store, "save", lambda config: (_ for _ in ()).throw(OSError("disk full")))

    response = client.post(
        "/client/priority-map/models/nodes/add/",
        {"name": "desktop", "host": "192.168.1.20", "instruction_port": 9000},
    )

    assert response.status_code == 500
    assert b"Could not save remote computer: disk full" in response.content
    assert "desktop" not in runtime.config_store.load().nodes


def test_remote_node_test_endpoint_reports_bounded_reachability(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    config = runtime.config_store.load()
    config.nodes["desktop"] = NodeConfig("remote", "192.168.1.20", 9000)
    runtime.config_store.save(config)
    calls = remote_health(monkeypatch, reachable=True)

    reachable = client.post("/client/priority-map/models/nodes/desktop/test/")
    monkeypatch.setattr(
        "web.views.InstructionClient", lambda *args, **kwargs: type("Client", (), {"health": lambda self: False})()
    )
    unreachable = client.post("/client/priority-map/models/nodes/desktop/test/")

    assert reachable.json() == {"state": "reachable", "message": "Instruction server is reachable."}
    assert unreachable.json() == {"state": "unreachable", "message": "Instruction server is unreachable."}
    assert calls == [("192.168.1.20", 9000, 2.0, 0)]
    assert client.post("/client/priority-map/models/nodes/missing/test/").status_code == 404
    assert client.post("/client/priority-map/models/nodes/local/test/").status_code == 400


def test_edit_remote_node_renames_service_references_and_preserves_extras(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    remote_health(monkeypatch, reachable=True)
    config = runtime.config_store.load()
    config.nodes["desktop"] = NodeConfig("remote", "192.168.1.20", 9000, {"node_extension": {"keep": True}})
    runtime.config_store.save(config)
    client.post("/services/llm/start/", llm_post_data(node="desktop"))

    response = client.post(
        "/client/priority-map/models/nodes/desktop/edit/",
        {"name": "gpu-desktop", "host": "192.168.1.21", "instruction_port": 9001},
    )
    saved = runtime.config_store.load()

    assert response.status_code == 302
    assert "desktop" not in saved.nodes
    assert saved.nodes["gpu-desktop"].host == "192.168.1.21"
    assert saved.nodes["gpu-desktop"].extra == {"node_extension": {"keep": True}}
    assert saved.priority_map.services["llm"].node == "gpu-desktop"


def test_edit_remote_node_rejects_duplicate_and_supports_save_anyway(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    config = runtime.config_store.load()
    config.nodes["desktop"] = NodeConfig("remote", "192.168.1.20", 9000)
    config.nodes["gpu"] = NodeConfig("remote", "192.168.1.21", 9001)
    runtime.config_store.save(config)
    remote_health(monkeypatch, reachable=False)

    duplicate = client.post(
        "/client/priority-map/models/nodes/desktop/edit/",
        {"name": "gpu", "host": "192.168.1.20", "instruction_port": 9000},
    )
    first = client.post(
        "/client/priority-map/models/nodes/desktop/edit/",
        {"name": "desktop", "host": "192.168.1.22", "instruction_port": 9002},
    )
    saved = client.post(
        "/client/priority-map/models/nodes/desktop/edit/",
        {"name": "desktop", "host": "192.168.1.22", "instruction_port": 9002, "save_anyway": "1"},
    )

    assert duplicate.status_code == 400
    assert b"already exists" in duplicate.content
    assert first.status_code == 400
    assert b"Save anyway" in first.content
    assert b"desktop" in first.content and b"gpu" in first.content
    assert saved.status_code == 302
    assert runtime.config_store.load().nodes["desktop"].host == "192.168.1.22"


def test_remote_node_deletion_enforces_dependencies_and_post_csrf(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    config = runtime.config_store.load()
    config.nodes["desktop"] = NodeConfig("remote", "192.168.1.20", 9000)
    runtime.config_store.save(config)
    remote_health(monkeypatch, reachable=True)
    client.post("/services/llm/start/", llm_post_data(node="desktop"))

    referenced = client.post("/client/priority-map/models/nodes/desktop/delete/")
    local = client.post("/client/priority-map/models/nodes/local/delete/")
    get_delete = client.get("/client/priority-map/models/nodes/desktop/delete/")
    csrf_client = Client(enforce_csrf_checks=True)
    csrf = csrf_client.post("/client/priority-map/models/nodes/desktop/delete/")
    config = runtime.config_store.load()
    del config.priority_map.services["llm"]
    runtime.config_store.save(config)
    deleted = client.post("/client/priority-map/models/nodes/desktop/delete/")

    assert referenced.status_code == 409
    assert b"used by LLM" in referenced.content
    assert local.status_code == 400
    assert get_delete.status_code == 405
    assert csrf.status_code == 403
    assert deleted.status_code == 302
    assert "desktop" not in runtime.config_store.load().nodes


def test_repository_inspection_filters_split_shards(client, monkeypatch, runtime) -> None:
    monkeypatch.setattr("web.views.get_runtime", lambda: runtime)
    monkeypatch.setattr(
        "web.views.LLMRuntime.list_repository_files",
        lambda _repo: ["model.gguf", "model-00001-of-00002.gguf", "mmproj.gguf"],
    )
    response = client.post(
        "/client/priority-map/models/llm/inspect-repository/",
        data='{"hf_repo":"org/model"}',
        content_type="application/json",
    )
    assert response.status_code == 200
    assert response.json()["models"] == ["model.gguf"]
    assert not response.json()["model_ambiguous"]
