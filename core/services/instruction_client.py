from __future__ import annotations

import time
from typing import IO, Any, Callable

import requests

from core.errors import InstructionError
from core.services.specs import ServiceEndpoint, ServiceSpec, ServiceStatus


class InstructionClient:
    """Synchronous bounded client for the remote control plane."""

    def __init__(
        self,
        host: str,
        port: int,
        timeout: float = 30.0,
        retries: int = 2,
        service_start_timeout: float = 660.0,
        artifact_timeout: float = 3600.0,
    ) -> None:
        if isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError("Instruction port must be between 1 and 65535.")
        self.host = host.strip().rstrip("/")
        self.base_url = f"http://{self.host}:{port}"
        self.timeout = timeout
        self.retries = max(0, min(retries, 3))
        self.service_start_timeout = max(float(timeout), float(service_start_timeout))
        self.artifact_timeout = max(float(timeout), float(artifact_timeout))

    def health(self) -> bool:
        try:
            return self._request("GET", "/health").status_code == 200
        except InstructionError:
            return False

    def start_service(self, spec: ServiceSpec) -> ServiceEndpoint:
        payload = self._json(
            self._request(
                "POST",
                "/services/start",
                json=spec.to_dict(),
                timeout=self.service_start_timeout,
            )
        )
        try:
            return ServiceEndpoint.from_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise InstructionError(f"Instruction server returned an invalid endpoint: {exc}") from exc

    def ensure_service(self, spec: ServiceSpec) -> ServiceEndpoint:
        """Reuse an identical running remote service, otherwise start/replace it."""
        for status in self.list_services():
            if (
                status.port == spec.port
                and status.running
                and status.service_type == spec.service_type
                and status.settings == spec.settings
            ):
                return ServiceEndpoint(
                    host=str(spec.settings.get("bind_host", self.host)),
                    port=spec.port,
                    service_type=spec.service_type,
                )
        return self.start_service(spec)

    def stop_service(self, port: int) -> None:
        self._request("POST", "/services/stop", json={"port": port})

    def upload_sam_checkpoint(self, file_handle: IO[bytes], *, filename: str, size: int) -> dict[str, Any]:
        """Stream one checkpoint to the selected remote compute node without buffering it in memory."""
        if isinstance(size, bool) or size < 0:
            raise ValueError("Checkpoint size must be a non-negative integer.")
        try:
            file_handle.seek(0)
        except (AttributeError, OSError):
            pass
        try:
            response = requests.post(
                f"{self.base_url}/artifacts/sam3/checkpoint",
                data=file_handle,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(size),
                    "X-Arcadia-Filename": filename,
                },
                timeout=self.artifact_timeout,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise InstructionError(self._error_message(response)) from exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise InstructionError(f"SAM3 checkpoint upload failed: {exc}") from exc
        payload = self._json(response)
        if not isinstance(payload, dict) or not isinstance(payload.get("path"), str):
            raise InstructionError("Instruction server returned an invalid checkpoint upload response.")
        return payload

    def list_services(self) -> list[ServiceStatus]:
        payload = self._json(self._request("GET", "/services"))
        if not isinstance(payload, list):
            raise InstructionError("Instruction server returned an invalid service list.")
        return [ServiceStatus.from_dict(item) for item in payload]

    def get_logs(self, port: int, tail: int = 200) -> str:
        return self._request("GET", f"/services/{port}/logs", params={"tail": tail}).text

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self.timeout)
        request_method: Callable[..., requests.Response] = getattr(requests, method.lower())
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = request_method(f"{self.base_url}{path}", **kwargs)
                if response.status_code in (502, 503, 504) and attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response
            except requests.HTTPError as exc:
                raise InstructionError(self._error_message(response)) from exc
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
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
