from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ServiceType = Literal["llm", "sam3"]


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """Serializable desired configuration for one inference service port."""

    service_type: ServiceType
    port: int
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.service_type not in ("llm", "sam3"):
            raise ValueError("service_type must be 'llm' or 'sam3'.")
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
    def from_dict(cls, data: dict[str, Any]) -> ServiceSpec:
        if not isinstance(data, dict):
            raise TypeError("Service specification must be a dictionary.")
        missing = [key for key in ("service_type", "port") if key not in data]
        if missing:
            raise ValueError(f"Missing required service field: {missing[0]}")
        return cls(
            service_type=data["service_type"],
            port=data["port"],
            settings=data.get("settings", {}),
        )


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    """Direct data-plane address of a running inference service."""

    host: str
    port: int
    service_type: ServiceType
    scheme: str = "http"

    def __post_init__(self) -> None:
        host = self.host.strip()
        scheme = self.scheme.strip().lower()
        if not host:
            raise ValueError("Endpoint host cannot be empty.")
        if self.service_type not in ("llm", "sam3"):
            raise ValueError("service_type must be 'llm' or 'sam3'.")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise TypeError("Endpoint port must be an integer.")
        if not 1 <= self.port <= 65535:
            raise ValueError("Endpoint port must be between 1 and 65535.")
        if scheme not in ("http", "https"):
            raise ValueError("Endpoint scheme must be 'http' or 'https'.")
        object.__setattr__(self, "host", host.rstrip("/"))
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
    def from_dict(cls, data: dict[str, Any]) -> ServiceEndpoint:
        if not isinstance(data, dict):
            raise TypeError("Service endpoint must be a dictionary.")
        return cls(
            host=data["host"],
            port=data["port"],
            service_type=data["service_type"],
            scheme=data.get("scheme", "http"),
        )


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
    def from_dict(cls, data: dict[str, Any]) -> ServiceStatus:
        return cls(
            port=data["port"],
            service_type=data["service_type"],
            running=bool(data["running"]),
            settings=dict(data.get("settings", {})),
            log_path=str(data.get("log_path", "")),
        )
