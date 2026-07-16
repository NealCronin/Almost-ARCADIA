from __future__ import annotations

import time
from typing import Any, Callable

import requests

from core.errors import InstructionError
from core.services.specs import ServiceEndpoint, ServiceSpec, ServiceStatus


class InstructionClient:
    """Synchronous bounded client for the remote control plane."""

    def __init__(self, host: str, port: int, timeout: float = 30.0, retries: int = 2) -> None:
        if isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("Instruction port must be between 1 and 65535.")
        self.base_url = f"http://{host.strip().rstrip('/')}:{port}"
        self.timeout = timeout
        self.retries = max(0, min(retries, 3))

    def health(self) -> bool:
        try:
            response = self._request("GET", "/health")
            return response.status_code == 200
        except InstructionError:
            return False

    def start_service(self, spec: ServiceSpec) -> ServiceEndpoint:
        response = self._request("POST", "/services/start", json=spec.to_dict())
        payload = self._json(response)
        try:
            return ServiceEndpoint.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise InstructionError(f"Instruction server returned an invalid endpoint: {exc}") from exc

    def stop_service(self, port: int) -> None:
        self._request("POST", "/services/stop", json={"port": port})

    def list_services(self) -> list[ServiceStatus]:
        response = self._request("GET", "/services")
        payload = self._json(response)
        if not isinstance(payload, list):
            raise InstructionError("Instruction server returned an invalid service list.")
        try:
            return [ServiceStatus.from_dict(item) for item in payload]
        except (KeyError, TypeError, ValueError) as exc:
            raise InstructionError(f"Instruction server returned invalid service status: {exc}") from exc

    def get_logs(self, port: int, tail: int = 200) -> str:
        response = self._request("GET", f"/services/{port}/logs", params={"tail": tail})
        return response.text

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        request_method: Callable[..., requests.Response] = getattr(requests, method.lower())
        attempts = self.retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = request_method(f"{self.base_url}{path}", **kwargs)
                status = response.status_code if isinstance(response.status_code, int) else 200
                if status in (502, 503, 504) and attempt + 1 < attempts:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    raise InstructionError(self._error_message(response)) from exc
                return response
            except InstructionError:
                raise
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt + 1 < attempts:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                break
        raise InstructionError(f"Instruction request {method} {path} failed: {last_error}")

    @staticmethod
    def _json(response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise InstructionError("Instruction server returned invalid JSON.") from exc

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        except ValueError:
            detail = response.text
        return f"Instruction server HTTP {response.status_code}: {detail}"
