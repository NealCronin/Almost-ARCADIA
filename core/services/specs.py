from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from core.errors import ConfigurationError

ServiceType = Literal["llm", "visual_llm", "sam3"]


@dataclass(slots=True)
class ServiceSpec:
    service_type: ServiceType
    port: int
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.service_type not in ("llm", "visual_llm", "sam3"):
            raise ConfigurationError(f"Unsupported service type: {self.service_type!r}.")
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ConfigurationError("Service port must be an integer between 1 and 65535.")
        if not isinstance(self.settings, dict):
            raise ConfigurationError("Service settings must be an object.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_type": self.service_type,
            "port": self.port,
            "settings": copy.deepcopy(self.settings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceSpec":
        if not isinstance(data, dict):
            raise ConfigurationError("Service specification must be an object.")
        service_type = data.get("service_type")
        port = data.get("port")
        if service_type not in ("llm", "visual_llm", "sam3"):
            raise ConfigurationError(f"Unsupported service type: {service_type!r}.")
        if isinstance(port, bool) or not isinstance(port, int):
            raise ConfigurationError("Service port must be an integer.")
        return cls(
            service_type=cast(ServiceType, service_type),
            port=port,
            settings=copy.deepcopy(data.get("settings", {})),
        )


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    host: str
    port: int
    service_type: ServiceType

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "service_type": self.service_type}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceEndpoint":
        return cls(str(data["host"]), int(data["port"]), data["service_type"])


@dataclass(frozen=True, slots=True)
class ServiceStatus:
    port: int
    service_type: ServiceType
    running: bool
    settings: dict[str, Any]
    log_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "port": self.port,
            "service_type": self.service_type,
            "running": self.running,
            "settings": copy.deepcopy(self.settings),
            "log_path": self.log_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServiceStatus":
        return cls(
            port=int(data["port"]),
            service_type=data["service_type"],
            running=bool(data["running"]),
            settings=copy.deepcopy(data.get("settings", {})),
            log_path=str(data.get("log_path", "")),
        )
