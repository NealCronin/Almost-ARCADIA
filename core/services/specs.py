from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ServiceType = Literal["llm", "sam3"]


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """Description of a model service that should run on one port."""

    service_type: ServiceType
    port: int
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.service_type not in ("llm", "sam3"):
            raise ValueError(
                f"Unsupported service type: {self.service_type!r}. "
                "Expected 'llm' or 'sam3'."
            )
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
            raise TypeError("Service specification must be a dictionary.")
        try:
            service_type = data["service_type"]
            port = data["port"]
        except KeyError as exc:
            raise ValueError(
                f"Missing required service field: {exc.args[0]}"
            ) from exc

        return cls(
            service_type=service_type,
            port=port,
            settings=data.get("settings", {}),
        )


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    """Address of a running inference service."""

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
            raise ValueError(f"Unsupported service type: {self.service_type!r}.")
        if isinstance(self.port, bool) or not isinstance(self.port, int):
            raise TypeError("Endpoint port must be an integer.")
        if not 1 <= self.port <= 65535:
            raise ValueError("Endpoint port must be between 1 and 65535.")
        if scheme not in ("http", "https"):
            raise ValueError("Endpoint scheme must be 'http' or 'https'.")

        object.__setattr__(self, "host", host)
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
        return cls(
            host=data["host"],
            port=data["port"],
            service_type=data["service_type"],
            scheme=data.get("scheme", "http"),
        )
