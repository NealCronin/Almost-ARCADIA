import pytest

from core.services.specs import ServiceEndpoint, ServiceSpec


def test_service_spec_round_trip() -> None:
    original = ServiceSpec(
        service_type="llm",
        port=8081,
        settings={"hf_repo": "org/model", "hf_file": "model.gguf"},
    )
    assert ServiceSpec.from_dict(original.to_dict()) == original


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_service_spec_rejects_invalid_port(port: int) -> None:
    with pytest.raises(ValueError):
        ServiceSpec(service_type="llm", port=port)


def test_endpoint_base_url() -> None:
    endpoint = ServiceEndpoint(
        host="127.0.0.1",
        port=8081,
        service_type="llm",
    )
    assert endpoint.base_url == "http://127.0.0.1:8081"
