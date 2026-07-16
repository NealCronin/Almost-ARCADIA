from fastapi.testclient import TestClient

from core.services.instruction_server import create_app
from core.services.specs import ServiceEndpoint, ServiceStatus


class FakeController:
    public_host = "100.96.40.81"

    def __init__(self) -> None:
        self.started = []
        self.stopped = []

    def start(self, spec):
        self.started.append(spec)
        return ServiceEndpoint(self.public_host, spec.port, spec.service_type)

    def stop(self, port):
        self.stopped.append(port)

    def stop_all(self):
        return None

    def list_services(self):
        return [ServiceStatus(8081, "llm", True, {}, "llm.log")]

    def get_logs(self, port, tail=200):
        return f"port={port} tail={tail}"


def valid_settings() -> dict[str, object]:
    return {"hf_repo": "org/model", "bind_host": "100.96.40.81", "models_cache_subdir": "huggingface"}


def test_instruction_server_accepts_declarative_matching_llm() -> None:
    controller = FakeController()
    client = TestClient(create_app(controller))
    started = client.post("/services/start", json={"service_type": "llm", "port": 8081, "settings": valid_settings()})
    assert started.status_code == 200
    assert controller.started[0].settings["bind_host"] == "100.96.40.81"
    assert client.post("/services/stop", json={"port": 8081}).json()["stopped"] is True


def test_instruction_server_rejects_command_cache_and_bind_mismatch() -> None:
    client = TestClient(create_app(FakeController()))
    for settings in (
        {"command": ["cmd.exe"]},
        {**valid_settings(), "bind_host": "127.0.0.1"},
        {**valid_settings(), "hf_cache_dir": "/tmp/cache"},
    ):
        assert (
            client.post("/services/start", json={"service_type": "llm", "port": 8081, "settings": settings}).status_code
            == 422
        )
