from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.errors import ConfigurationError
from core.services.specs import ServiceSpec

NodeMode = Literal["local", "remote"]


@dataclass(slots=True)
class NodeConfig:
    mode: NodeMode
    host: str
    instruction_port: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("local", "remote"):
            raise ConfigurationError("Node mode must be 'local' or 'remote'.")
        self.host = self.host.strip()
        if not self.host:
            raise ConfigurationError("Node host cannot be empty.")
        if self.instruction_port is not None and not 1 <= self.instruction_port <= 65535:
            raise ConfigurationError("Instruction port must be between 1 and 65535.")
        if self.mode == "remote" and self.instruction_port is None:
            raise ConfigurationError("Remote nodes require instruction_port.")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"mode": self.mode, "host": self.host}
        if self.instruction_port is not None:
            result["instruction_port"] = self.instruction_port
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeConfig:
        if not isinstance(data, dict):
            raise ConfigurationError("Node configuration must be an object.")
        return cls(
            mode=data.get("mode", "local"),
            host=data.get("host", "127.0.0.1"),
            instruction_port=data.get("instruction_port"),
        )


@dataclass(slots=True)
class ConfiguredService:
    """A desired service plus the node where it should run."""

    node: str
    spec: ServiceSpec

    @property
    def service_type(self):
        return self.spec.service_type

    @property
    def port(self) -> int:
        return self.spec.port

    @property
    def settings(self) -> dict[str, Any]:
        return self.spec.settings

    def to_dict(self) -> dict[str, Any]:
        return {"node": self.node, **self.spec.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfiguredService:
        if not isinstance(data, dict):
            raise ConfigurationError("Service configuration must be an object.")
        return cls(
            node=str(data.get("node", "local")),
            spec=ServiceSpec.from_dict(data),
        )


@dataclass(slots=True)
class PipelineConfig:
    sam_step: int = 5
    run_at_source_fps: bool = False
    sam_resize: int | None = None
    task: str = "Find cars"
    debrief: str = ""
    prompts: list[str] = field(default_factory=list)
    sam_confidence: float = 0.25
    max_image_edge: int | None = 640
    debug: bool = False
    record: bool = True
    panoramic: bool = False
    graph_agent: bool = False
    gps_csv: str | None = None
    camera_intrinsics: str | None = None
    scene_model: str | None = None

    def __post_init__(self) -> None:
        if self.sam_step < 1:
            raise ConfigurationError("sam_step must be at least 1.")
        if self.sam_resize is not None and self.sam_resize < 1:
            raise ConfigurationError("sam_resize must be positive or null.")
        if not 0 <= self.sam_confidence <= 1:
            raise ConfigurationError("sam_confidence must be between 0 and 1.")
        self.prompts = [str(prompt).strip() for prompt in self.prompts if str(prompt).strip()]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PipelineConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigurationError("Pipeline configuration must be an object.")
        allowed = {
            "sam_step",
            "run_at_source_fps",
            "sam_resize",
            "task",
            "debrief",
            "prompts",
            "sam_confidence",
            "max_image_edge",
            "debug",
            "record",
            "panoramic",
            "graph_agent",
            "gps_csv",
            "camera_intrinsics",
            "scene_model",
        }
        values = {key: value for key, value in data.items() if key in allowed}
        return cls(**values)


@dataclass(slots=True)
class AppConfig:
    nodes: dict[str, NodeConfig] = field(default_factory=dict)
    services: dict[str, ConfiguredService] = field(default_factory=dict)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output_root: Path = Path("outputs")

    def __post_init__(self) -> None:
        if "local" not in self.nodes:
            self.nodes["local"] = NodeConfig(mode="local", host="127.0.0.1")
        for name in self.services:
            if self.services[name].node not in self.nodes:
                raise ConfigurationError(f"Service {name!r} references unknown node {self.services[name].node!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
            "services": {name: service.to_dict() for name, service in self.services.items()},
            "pipeline": self.pipeline.to_dict(),
            "output_root": str(self.output_root),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        if not isinstance(data, dict):
            raise ConfigurationError("Configuration root must be a JSON object.")
        nodes = {str(name): NodeConfig.from_dict(value) for name, value in dict(data.get("nodes", {})).items()}
        services = {
            str(name): ConfiguredService.from_dict(value) for name, value in dict(data.get("services", {})).items()
        }
        output_root = data.get("output_root", "outputs")
        if not isinstance(output_root, str) or not output_root.strip():
            raise ConfigurationError("output_root must be a non-empty path.")
        return cls(
            nodes=nodes,
            services=services,
            pipeline=PipelineConfig.from_dict(data.get("pipeline")),
            output_root=Path(output_root),
        )


class ConfigStore:
    """Load and atomically save one JSON configuration file."""

    def __init__(self, path: str | Path = "config.json") -> None:
        self.path = Path(path)

    def load(self) -> AppConfig:
        if not self.path.exists():
            return AppConfig()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Could not read configuration {self.path}: {exc}") from exc
        return AppConfig.from_dict(data)

    def save(self, config: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise
