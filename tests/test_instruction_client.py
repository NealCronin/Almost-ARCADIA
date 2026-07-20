from __future__ import annotations

from typing import Any

from core.services.instruction_client import InstructionClient
from core.services.specs import ServiceEndpoint, ServiceSpec, ServiceStatus


def spec() -> ServiceSpec:
    return ServiceSpec(
        "llm",
        8081,
        {
            "hf_repo": "owner/repo",
            "hf_revision": "main",
            "hf_file": "model.gguf",
            "bind_host": "10.0.0.20",
            "n_ctx": 4096,
            "vision_enabled": False,
            "temperature": 0.1,
            "max_tokens": 256,
            "extra_args": [],
            "models_cache_subdir": "huggingface",
        },
    )


def test_start_service_uses_long_model_start_timeout(monkeypatch):
    client = InstructionClient("10.0.0.20", 9000, timeout=5, service_start_timeout=700)
    observed: dict[str, Any] = {}

    class Response:
        def json(self):
            return {"host": "10.0.0.20", "port": 8081, "service_type": "llm"}

    def fake_request(method: str, path: str, **kwargs: Any):
        observed.update(kwargs)
        return Response()

    monkeypatch.setattr(client, "_request", fake_request)
    endpoint = client.start_service(spec())
    assert endpoint == ServiceEndpoint("10.0.0.20", 8081, "llm")
    assert observed["timeout"] == 700


def test_ensure_service_reuses_matching_remote_process(monkeypatch):
    client = InstructionClient("10.0.0.20", 9000)
    service_spec = spec()
    status = ServiceStatus(
        port=service_spec.port,
        service_type=service_spec.service_type,
        running=True,
        settings=service_spec.settings,
        log_path="logs/llm-8081.log",
    )
    monkeypatch.setattr(client, "list_services", lambda: [status])

    def should_not_start(_spec: ServiceSpec):
        raise AssertionError("matching service should be reused")

    monkeypatch.setattr(client, "start_service", should_not_start)
    endpoint = client.ensure_service(service_spec)
    assert endpoint == ServiceEndpoint("10.0.0.20", 8081, "llm")


def test_upload_sam_checkpoint_streams_raw_file(monkeypatch):
    import io

    client = InstructionClient("10.0.0.20", 9000, artifact_timeout=123)
    observed: dict[str, Any] = {}

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"path": "C:/arcadia/workspace/huggingface/models/sam3.pt", "size_bytes": 4}

    def fake_post(url: str, **kwargs: Any):
        observed["url"] = url
        observed.update(kwargs)
        observed["bytes"] = kwargs["data"].read()
        return Response()

    monkeypatch.setattr("core.services.instruction_client.requests.post", fake_post)
    payload = client.upload_sam_checkpoint(io.BytesIO(b"sam3"), filename="sam3.pt", size=4)

    assert payload["path"].endswith("sam3.pt")
    assert observed["url"] == "http://10.0.0.20:9000/artifacts/sam3/checkpoint"
    assert observed["headers"]["X-Arcadia-Filename"] == "sam3.pt"
    assert observed["headers"]["Content-Length"] == "4"
    assert observed["timeout"] == 123
    assert observed["bytes"] == b"sam3"
