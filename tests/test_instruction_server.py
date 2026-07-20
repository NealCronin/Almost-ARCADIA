from __future__ import annotations

from fastapi.testclient import TestClient

from core.errors import ServiceStartupError
from core.services.instruction_server import create_app


class Controller:
    def list_services(self):
        return []

    def start(self, spec):
        raise ServiceStartupError("llama-server binary is missing")


def test_remote_startup_error_is_returned_as_detail_not_internal_server_error():
    client = TestClient(create_app(Controller(), public_host="10.0.0.20"), raise_server_exceptions=False)
    response = client.post(
        "/services/start",
        json={
            "service_type": "llm",
            "port": 8081,
            "settings": {
                "hf_repo": "owner/repo",
                "hf_revision": "main",
                "hf_file": "model.gguf",
                "bind_host": "10.0.0.20",
                "n_ctx": 4096,
                "vision_enabled": False,
                "temperature": 0.1,
                "max_tokens": 256,
                "extra_args": [],
            },
        },
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "llama-server binary is missing"


def test_instruction_server_streams_sam_checkpoint_into_models_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCADIA_MODELS_DIR", str(tmp_path / "huggingface"))
    client = TestClient(create_app(Controller(), public_host="10.0.0.20"), raise_server_exceptions=False)

    response = client.post(
        "/artifacts/sam3/checkpoint",
        content=b"checkpoint-bytes",
        headers={
            "Content-Type": "application/octet-stream",
            "X-Arcadia-Filename": "sam3.pt",
        },
    )

    assert response.status_code == 200
    saved = tmp_path / "huggingface" / "models" / "sam3.pt"
    assert saved.read_bytes() == b"checkpoint-bytes"
    assert response.json()["path"] == str(saved.resolve())
