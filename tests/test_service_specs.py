from __future__ import annotations

import pytest

from core.services.specs import ServiceEndpoint, ServiceSpec


def test_spec_defensively_copies_settings() -> None:
    settings = {"model_path": "model.gguf"}
    spec = ServiceSpec("llm", 8081, settings)
    settings["changed"] = True
    assert "changed" not in spec.settings


@pytest.mark.parametrize("service_type", ["other", "SAM", ""])
def test_spec_rejects_invalid_type(service_type: str) -> None:
    with pytest.raises(ValueError):
        ServiceSpec(service_type, 8081)


def test_endpoint_normalizes_host_and_scheme() -> None:
    endpoint = ServiceEndpoint("  localhost/ ", 8081, "llm", "HTTP")
    assert endpoint.base_url == "http://localhost:8081"


def test_round_trip() -> None:
    spec = ServiceSpec("sam3", 8090, {"checkpoint": "weights.pt"})
    assert ServiceSpec.from_dict(spec.to_dict()) == spec
