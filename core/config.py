from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass, field
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
    extra: dict[str, Any] = field(default_factory=dict)

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
        result = copy.deepcopy(self.extra)
        result.update({"mode": self.mode, "host": self.host})
        if self.instruction_port is not None:
            result["instruction_port"] = self.instruction_port
        else:
            result.pop("instruction_port", None)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeConfig:
        if not isinstance(data, dict):
            raise ConfigurationError("Node configuration must be an object.")
        known = {"mode", "host", "instruction_port"}
        return cls(
            data.get("mode", "local"),
            data.get("host", "127.0.0.1"),
            data.get("instruction_port"),
            copy.deepcopy({key: value for key, value in data.items() if key not in known}),
        )


@dataclass(slots=True)
class ConfiguredService:
    """A desired service plus the node where it should run."""

    node: str
    spec: ServiceSpec
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def service_type(self) -> str:
        return self.spec.service_type

    @property
    def port(self) -> int:
        return self.spec.port

    @property
    def settings(self) -> dict[str, Any]:
        return self.spec.settings

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        result.update({"node": self.node, **self.spec.to_dict()})
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfiguredService:
        if not isinstance(data, dict):
            raise ConfigurationError("Service configuration must be an object.")
        known = {"node", "service_type", "port", "settings"}
        return cls(
            str(data.get("node", "local")),
            ServiceSpec.from_dict(data),
            copy.deepcopy({key: value for key, value in data.items() if key not in known}),
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
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sam_step < 1:
            raise ConfigurationError("sam_step must be at least 1.")
        if self.sam_resize is not None and self.sam_resize < 1:
            raise ConfigurationError("sam_resize must be positive or null.")
        if not 0 <= self.sam_confidence <= 1:
            raise ConfigurationError("sam_confidence must be between 0 and 1.")
        self.prompts = [str(prompt).strip() for prompt in self.prompts if str(prompt).strip()]

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        result.update({name: getattr(self, name) for name in self.__dataclass_fields__ if name != "extra"})
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PipelineConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigurationError("Pipeline configuration must be an object.")
        known = set(cls.__dataclass_fields__) - {"extra"}
        return cls(
            **{key: value for key, value in data.items() if key in known},
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known}),
        )


@dataclass(slots=True)
class PriorityMapOutputConfig:
    root: Path = Path("outputs")
    preview: Literal["mjpeg"] = "mjpeg"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.root).strip():
            raise ConfigurationError("Priority Map output root must be a non-empty path.")
        if self.preview != "mjpeg":
            raise ConfigurationError("Priority Map preview must be 'mjpeg'.")

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        result.update({"root": str(self.root), "preview": self.preview})
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PriorityMapOutputConfig:
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigurationError("Priority Map output configuration must be an object.")
        root = data.get("root", "outputs")
        if not isinstance(root, str) or not root.strip():
            raise ConfigurationError("Priority Map output root must be a non-empty path.")
        preview = data.get("preview", "mjpeg")
        if preview != "mjpeg":
            raise ConfigurationError("Priority Map preview must be 'mjpeg'.")
        known = {"root", "preview"}
        return cls(Path(root), preview, copy.deepcopy({key: value for key, value in data.items() if key not in known}))


@dataclass(slots=True)
class PriorityMapToolConfig:
    services: dict[str, ConfiguredService] = field(default_factory=dict)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output: PriorityMapOutputConfig = field(default_factory=PriorityMapOutputConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        result.update(
            {
                "services": {name: service.to_dict() for name, service in self.services.items()},
                "pipeline": self.pipeline.to_dict(),
                "output": self.output.to_dict(),
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PriorityMapToolConfig:
        if not isinstance(data, dict):
            raise ConfigurationError("Priority Map tool configuration must be an object.")
        services = data.get("services", {})
        if not isinstance(services, dict):
            raise ConfigurationError("Priority Map services must be an object.")
        known = {"services", "pipeline", "output"}
        return cls(
            services={str(name): ConfiguredService.from_dict(value) for name, value in services.items()},
            pipeline=PipelineConfig.from_dict(data.get("pipeline")),
            output=PriorityMapOutputConfig.from_dict(data.get("output")),
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known}),
        )


@dataclass(slots=True)
class AppConfig:
    nodes: dict[str, NodeConfig] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=lambda: {"priority-map": PriorityMapToolConfig()})
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if "local" not in self.nodes:
            self.nodes["local"] = NodeConfig(mode="local", host="127.0.0.1")
        if "priority-map" not in self.tools:
            self.tools["priority-map"] = PriorityMapToolConfig()
        priority_map = self.tools["priority-map"]
        if not isinstance(priority_map, PriorityMapToolConfig):
            raise ConfigurationError("Priority Map tool configuration must be an object.")
        for name, service in priority_map.services.items():
            if service.node not in self.nodes:
                raise ConfigurationError(f"Service {name!r} references unknown node {service.node!r}.")

    @property
    def priority_map(self) -> PriorityMapToolConfig:
        return self.tools["priority-map"]

    def to_dict(self) -> dict[str, Any]:
        tools = copy.deepcopy({name: value for name, value in self.tools.items() if name != "priority-map"})
        tools["priority-map"] = self.priority_map.to_dict()
        result = copy.deepcopy(self.extra)
        result.update(
            {
                "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
                "tools": tools,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        if not isinstance(data, dict):
            raise ConfigurationError("Configuration root must be a JSON object.")
        raw_nodes = data.get("nodes", {})
        if not isinstance(raw_nodes, dict):
            raise ConfigurationError("Nodes configuration must be an object.")
        nodes = {str(name): NodeConfig.from_dict(value) for name, value in raw_nodes.items()}
        known_root = {"nodes", "tools", "services", "pipeline", "output_root"}
        if "tools" in data:
            raw_tools = data["tools"]
            if not isinstance(raw_tools, dict):
                raise ConfigurationError("Tools configuration must be an object.")
            raw_priority_map = raw_tools.get("priority-map")
            if not isinstance(raw_priority_map, dict):
                raise ConfigurationError("tools.priority-map must be an object.")
            tools = {name: copy.deepcopy(value) for name, value in raw_tools.items() if name != "priority-map"}
            tools["priority-map"] = PriorityMapToolConfig.from_dict(raw_priority_map)
        else:
            output_root = data.get("output_root", "outputs")
            if not isinstance(output_root, str) or not output_root.strip():
                raise ConfigurationError("output_root must be a non-empty path.")
            raw_services = data.get("services", {})
            if not isinstance(raw_services, dict):
                raise ConfigurationError("Services configuration must be an object.")
            tools = {
                "priority-map": PriorityMapToolConfig(
                    services={str(name): ConfiguredService.from_dict(value) for name, value in raw_services.items()},
                    pipeline=PipelineConfig.from_dict(data.get("pipeline")),
                    output=PriorityMapOutputConfig(Path(output_root)),
                )
            }
        return cls(
            nodes=nodes,
            tools=tools,
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known_root}),
        )


class ConfigStore:
    """Load and atomically save one JSON configuration file."""

    def __init__(self, path: str | Path = "config.json", default_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.default_path = (
            Path(default_path) if default_path is not None else self.path.with_name("default_config.json")
        )

    def load(self) -> AppConfig:
        if not self.path.exists():
            if not self.default_path.exists():
                return AppConfig()
            try:
                payload = self.default_path.read_text(encoding="utf-8")
                config = AppConfig.from_dict(json.loads(payload))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigurationError(f"Could not read default configuration {self.default_path}: {exc}") from exc
            self._write_payload(payload)
            return config
        try:
            return AppConfig.from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Could not read configuration {self.path}: {exc}") from exc

    def save(self, config: AppConfig) -> None:
        self._write_payload(json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n")

    def _write_payload(self, payload: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
