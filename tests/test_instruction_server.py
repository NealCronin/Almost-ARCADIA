from fastapi.testclient import TestClient

from core.services.instruction_server import create_app
from core.services.specs import ServiceEndpoint, ServiceStatus


class FakeController:
    def __init__(self) -> None:
        self.started = []
        self.stopped = []

    def start(self, spec):
        self.started.append(spec)
        return ServiceEndpoint("10.0.0.2", spec.port, spec.service_type)

    def stop(self, port):
        self.stopped.append(port)

    def stop_all(self):
        return None

    def list_services(self):
        return [ServiceStatus(8081, "llm", True, {}, "llm.log")]

    def get_logs(self, port, tail=200):
        return f"port={port} tail={tail}"


def test_instruction_server_contract() -> None:
    controller = FakeController()
    client = TestClient(create_app(controller))
    assert client.get("/health").status_code == 200
    started = client.post(
        "/services/start", json={"service_type": "llm", "port": 8081, "settings": {"model_path": "m.gguf"}}
    )
    assert started.status_code == 200
    assert started.json()["host"] == "10.0.0.2"
    assert client.get("/services").json()[0]["port"] == 8081
    assert client.get("/services/8081/logs").text.startswith("port=8081")
    assert client.post("/services/stop", json={"port": 8081}).json()["stopped"] is True


def test_instruction_server_rejects_arbitrary_commands() -> None:
    client = TestClient(create_app(FakeController()))
    response = client.post(
        "/services/start", json={"service_type": "llm", "port": 8081, "settings": {"command": ["cmd.exe"]}}
    )
    assert response.status_code == 422
