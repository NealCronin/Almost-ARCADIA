from __future__ import annotations

import requests

from .specs import ServiceEndpoint, ServiceSpec


class InstructionClient:
    def __init__(self, host: str, port: int, timeout: float = 30.0) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def start_service(self, spec: ServiceSpec) -> ServiceEndpoint:
        response = requests.post(
            f"{self.base_url}/services/start",
            json=spec.to_dict(),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return ServiceEndpoint.from_dict(response.json())

    def stop_service(self, port: int) -> None:
        response = requests.post(
            f"{self.base_url}/services/stop",
            json={"port": port},
            timeout=self.timeout,
        )
        response.raise_for_status()

    def list_services(self) -> list[dict[str, object]]:
        response = requests.get(
            f"{self.base_url}/services",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
