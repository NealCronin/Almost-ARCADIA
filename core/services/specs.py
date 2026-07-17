from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ServiceType = Literal["llm", "visual_llm", "sam3"]


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """Serializable desired configuration for one inference service port."""

    service_type: ServiceType
    port: int
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.service_type not in ("llm", "visual_llm", "sam3"):
            raise ValueError("service_type must be 'llm', 'visual_llm', or 'sam3'.")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise TypeError("Service port must be an integer.")
        if not 1 <= self.port <= 65535:
            raise ValueError("Service port must be between 1 and 65535.")
        if not isinstance(self.settings, dict):
            raise TypeError("Service settings must be a dictionary.")
        object.__setattr__(self, "settings", dict(self.settings))

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_type": self.service_type,
            "port": self.port,
            "settings": dict(self.settings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceSpec":
        if not isinstance(data, dict):
            raise TypeError("ServiceSpec.from_dict requires a dict")
        service_type = data.get("service_type", "llm")
        if service_type not in ("llm", "visual_llm", "sam3"):
            raise ValueError(f"Unknown service type: {service_type!r}")
        port = data.get("port", 8081)
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError(f"Invalid port: {port!r}")
        settings = data.get("settings", {})
        if not isinstance(settings, dict):
            raise ValueError("settings must be a dict")
        return cls(service_type=service_type, port=port, settings=dict(settings))


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    """Direct data-plane address of a running inference service."""

    host: str
    port: int
    service_type: ServiceType
    scheme: str = "http"

    def __post_init__(self) -> None:
        host = self.host.strip().rstrip("/")
        if not host:
            raise ValueError("Endpoint host cannot be empty.")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise TypeError("Endpoint port must be an integer.")
        if not 1 <= self.port <= 65535:
            raise ValueError("Endpoint port must be between 1 and 65535.")
        if self.service_type not in ("llm", "visual_llm", "sam3"):
            raise ValueError(f"Unknown service type: {self.service_type!r}")
        scheme = self.scheme.strip().lower()
        if scheme not in ("http", "https"):
            raise ValueError(f"Invalid scheme: {self.scheme!r}")
        object.__setattr__(self, "host", host)
        object.__setattr__(self, "port", self.port)
        object.__setattr__(self, "service_type", self.service_type)
        object.__setattr__(self, "scheme", scheme)

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "service_type": self.service_type,
            "scheme": self.scheme,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceEndpoint":
        if not isinstance(data, dict):
            raise TypeError("ServiceEndpoint.from_dict requires a dict")
        host = data.get("host", "127.0.0.1")
        port = data.get("port", 8081)
        service_type = data.get("service_type", "llm")
        if service_type not in ("llm", "visual_llm", "sam3"):
            raise ValueError(f"Unknown service type: {service_type!r}")
        scheme = data.get("scheme", "http")
        return cls(host=host, port=port, service_type=service_type, scheme=scheme)


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    """Small status payload shared by the instruction API and UI."""

    port: int
    service_type: ServiceType
    running: bool
    settings: dict[str, Any] = field(default_factory=dict)
    log_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "service_type": self.service_type,
            "running": self.running,
            "settings": dict(self.settings),
            "log_path": self.log_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceStatus":
        return cls(
            port=data["port"],
            service_type=data["service_type"],
            running=bool(data["running"]),
            settings=dict(data.get("settings", {})),
            log_path=str(data.get("log_path", "")),
        )
