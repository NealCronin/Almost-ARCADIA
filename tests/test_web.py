from __future__ import annotations

import pytest
from django.test import Client, override_settings

from core.config import ConfigStore


@pytest.fixture
def config_path(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    monkeypatch.setenv("ARCADIA_CONFIG", str(path))
    # Runtime is created lazily; reset it after changing the environment.
    from web.runtime import set_runtime

    set_runtime(None)
    yield path
    set_runtime(None)


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_pages_render(config_path):
    client = Client()
    for url in ("/", "/client/", "/host/", "/client/priority-map/models/", "/analysis/", "/results/"):
        assert client.get(url).status_code == 200


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_models_page_has_only_simplified_llm_fields(config_path):
    body = Client().get("/client/priority-map/models/").content.decode()
    for label in (
        "Compute node",
        "Inference port",
        "Inference IP",
        "Hugging Face model",
        "Enable vision",
        "Context size",
        "Max output tokens",
        "Temperature",
        "Additional llama-server arguments",
    ):
        assert label in body
    assert "GPU layers" not in body
    assert "K-cache type" not in body
    assert "Draft model" not in body


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_save_llm_settings(config_path):
    response = Client().post(
        "/services/llm/start/",
        {
            "node": "local",
            "inference_port": "8081",
            "bind_host": "127.0.0.1",
            "hf_source": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/blob/main/Qwen3.5-2B-IQ4_XS.gguf",
            "n_ctx": "32768",
            "max_tokens": "512",
            "temperature": "0.1",
            "additional_arguments": "-ngl all, -fa on",
        },
    )
    assert response.status_code == 302
    config = ConfigStore(config_path).load()
    saved = config.priority_map.services["llm"]
    assert saved.settings["hf_file"] == "Qwen3.5-2B-IQ4_XS.gguf"
    assert saved.settings["extra_args"] == ["-ngl", "all", "-fa", "on"]
    assert "model_alias" not in saved.settings


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_owned_argument_is_rejected_in_form(config_path):
    response = Client().post(
        "/services/llm/start/",
        {
            "node": "local",
            "inference_port": "8081",
            "bind_host": "127.0.0.1",
            "hf_source": "owner/repo",
            "n_ctx": "32768",
            "max_tokens": "512",
            "temperature": "0.1",
            "additional_arguments": "--host 0.0.0.0",
        },
    )
    assert response.status_code == 400
    assert b"cannot override" in response.content


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_llm_test_chat_returns_plain_text(config_path, monkeypatch):
    from core.config import ConfiguredService, NodeConfig
    from core.inference.results import LLMResult
    from core.services.specs import ServiceEndpoint, ServiceSpec
    from web import views

    node = NodeConfig("local", "127.0.0.1")
    spec = ServiceSpec(
        "llm",
        8081,
        {
            "hf_repo": "owner/repo",
            "bind_host": "127.0.0.1",
            "n_ctx": 32768,
            "temperature": 0.1,
            "max_tokens": 512,
        },
    )
    configured = ConfiguredService("local", spec)
    endpoint = ServiceEndpoint("127.0.0.1", 8081, "llm")

    monkeypatch.setattr(
        views,
        "_resolve_llm",
        lambda _config, _role: ("llm", configured, node, spec),
    )
    monkeypatch.setattr(views, "_ensure_endpoint", lambda _node, _spec: endpoint)
    monkeypatch.setattr(
        views.LLMClient,
        "chat",
        lambda self, prompt, images=None: LLMResult("Hello", {"choices": []}),
    )

    response = Client().post(
        "/client/priority-map/models/llm/test-chat/",
        {"prompt": "Hi"},
    )

    assert response.status_code == 200
    assert response["Content-Type"].startswith("text/plain")
    assert response.content.decode() == "Hello"


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_llm_test_chat_uses_reasoning_fallback(config_path, monkeypatch):
    from core.config import ConfiguredService, NodeConfig
    from core.inference.results import LLMResult
    from core.services.specs import ServiceEndpoint, ServiceSpec
    from web import views

    node = NodeConfig("local", "127.0.0.1")
    spec = ServiceSpec(
        "llm",
        8081,
        {
            "hf_repo": "owner/repo",
            "bind_host": "127.0.0.1",
            "n_ctx": 32768,
            "temperature": 0.1,
            "max_tokens": 512,
        },
    )
    configured = ConfiguredService("local", spec)
    endpoint = ServiceEndpoint("127.0.0.1", 8081, "llm")
    raw = {
        "choices": [
            {
                "message": {"content": "", "reasoning_content": "Visible reasoning"},
                "finish_reason": "stop",
            }
        ]
    }

    monkeypatch.setattr(
        views,
        "_resolve_llm",
        lambda _config, _role: ("llm", configured, node, spec),
    )
    monkeypatch.setattr(views, "_ensure_endpoint", lambda _node, _spec: endpoint)
    monkeypatch.setattr(
        views.LLMClient,
        "chat",
        lambda self, prompt, images=None: LLMResult("", raw),
    )

    response = Client().post(
        "/client/priority-map/models/llm/test-chat/",
        {"prompt": "Hi"},
    )

    assert response.status_code == 200
    assert response.content.decode() == "Visible reasoning"


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_models_page_contains_sam3_image_test(config_path):
    body = Client().get("/client/priority-map/models/").content.decode()
    assert "Test SAM3" in body
    assert "Search term" in body
    assert "data-test-sam-image" in body
    assert "data-test-sam-result-image" in body


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_sam3_test_returns_segmented_png(config_path, monkeypatch):
    import cv2
    import numpy as np
    from django.core.files.uploadedfile import SimpleUploadedFile

    from core.config import ConfiguredService, NodeConfig
    from core.inference.results import SegmentationResult
    from core.services.specs import ServiceEndpoint, ServiceSpec
    from web import views

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (30, 40, 50)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[10:35, 18:48] = 1
    node = NodeConfig("local", "127.0.0.1")
    spec = ServiceSpec(
        "sam3",
        8090,
        {
            "bind_host": "127.0.0.1",
            "checkpoint": "sam3.pt",
            "confidence": 0.25,
            "extra_args": [],
        },
    )
    configured = ConfiguredService("local", spec)
    endpoint = ServiceEndpoint("127.0.0.1", 8090, "sam3")

    monkeypatch.setattr(views, "_resolve_sam", lambda _config: (configured, node, spec))
    monkeypatch.setattr(views, "_ensure_endpoint", lambda _node, _spec: endpoint)
    monkeypatch.setattr(
        views.SAMClient,
        "segment",
        lambda self, frame, prompts, confidence=0.25: SegmentationResult(
            masks=[mask.tolist()],
            labels=[prompts[0]],
            confidences=[0.91],
            bounding_boxes=[[18, 10, 48, 35]],
        ),
    )

    response = Client().post(
        "/client/priority-map/models/sam3/test/",
        {
            "search_term": "car",
            "image": SimpleUploadedFile("test.png", encoded.tobytes(), content_type="image/png"),
        },
    )

    assert response.status_code == 200
    assert response["Content-Type"] == "image/png"
    assert response["X-Arcadia-Segment-Count"] == "1"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
    decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_sam3_test_reports_no_segments(config_path, monkeypatch):
    import cv2
    import numpy as np
    from django.core.files.uploadedfile import SimpleUploadedFile

    from core.config import ConfiguredService, NodeConfig
    from core.inference.results import SegmentationResult
    from core.services.specs import ServiceEndpoint, ServiceSpec
    from web import views

    image = np.zeros((20, 20, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    node = NodeConfig("local", "127.0.0.1")
    spec = ServiceSpec(
        "sam3",
        8090,
        {
            "bind_host": "127.0.0.1",
            "checkpoint": "sam3.pt",
            "confidence": 0.25,
            "extra_args": [],
        },
    )
    configured = ConfiguredService("local", spec)
    endpoint = ServiceEndpoint("127.0.0.1", 8090, "sam3")

    monkeypatch.setattr(views, "_resolve_sam", lambda _config: (configured, node, spec))
    monkeypatch.setattr(views, "_ensure_endpoint", lambda _node, _spec: endpoint)
    monkeypatch.setattr(
        views.SAMClient,
        "segment",
        lambda self, frame, prompts, confidence=0.25: SegmentationResult([], [], [], []),
    )

    response = Client().post(
        "/client/priority-map/models/sam3/test/",
        {
            "search_term": "car",
            "image": SimpleUploadedFile("test.png", encoded.tobytes(), content_type="image/png"),
        },
    )

    assert response.status_code == 422
    assert "found no segments" in response.content.decode()


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_models_page_contains_sam3_checkpoint_browse(config_path):
    body = Client().get("/client/priority-map/models/").content.decode()
    assert "Browse…" in body
    assert "data-sam-checkpoint-file" in body
    assert "/client/priority-map/models/sam3/checkpoint/upload/" in body


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_local_sam3_checkpoint_upload_returns_compute_node_path(config_path, tmp_path, monkeypatch):
    from django.core.files.uploadedfile import SimpleUploadedFile

    monkeypatch.setenv("ARCADIA_MODELS_DIR", str(tmp_path / "huggingface"))
    response = Client().post(
        "/client/priority-map/models/sam3/checkpoint/upload/",
        {
            "node": "local",
            "checkpoint": SimpleUploadedFile("sam3.pt", b"checkpoint", content_type="application/octet-stream"),
        },
    )

    assert response.status_code == 201
    payload = response.json()
    path = tmp_path / "huggingface" / "models" / "sam3.pt"
    assert path.read_bytes() == b"checkpoint"
    assert payload["checkpoint"] == str(path.resolve())
    assert payload["node"] == "local"


@override_settings(ALLOWED_HOSTS=["testserver"])
def test_host_page_renders_and_saves_sam3_checkpoint_controls(config_path, tmp_path, monkeypatch):
    huggingface = tmp_path / "huggingface"
    monkeypatch.setenv("ARCADIA_HUGGINGFACE_DIR", str(huggingface))
    checkpoint = huggingface / "models" / "sam3.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    body = Client().get("/host/").content.decode()
    assert "Applied device" in body
    assert "Restart required" in body
    assert "SAM3 checkpoint" in body
    assert "Browse or upload" in body
    assert "Save checkpoint" in body

    response = Client().post("/host/sam3/checkpoint/save/", {"checkpoint": str(checkpoint)})

    assert response.status_code == 302
    assert ConfigStore(config_path).load().host_listener.sam3_checkpoint == str(checkpoint.resolve())


def test_host_restart_requirement_compares_checkpoint_and_device():
    from core.services.specs import ServiceStatus
    from web.views import _sam_restart_required

    running = ServiceStatus(
        port=8090,
        service_type="sam3",
        running=True,
        settings={"checkpoint": "/host/huggingface/models/sam3.pt", "device": "mps"},
        log_path="",
    )

    assert not _sam_restart_required(running, "/host/huggingface/models/sam3.pt", "mps")
    assert _sam_restart_required(running, "/host/huggingface/models/new.pt", "mps")
    assert _sam_restart_required(running, "/host/huggingface/models/sam3.pt", "cpu")
