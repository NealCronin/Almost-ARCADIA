from __future__ import annotations

import copy
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from core.errors import ConfigurationError
from core.networking import validate_ipv4
from core.services.specs import ServiceSpec

NodeMode = Literal["local", "remote"]


@dataclass(slots=True)
class HostListenerConfig:
    host: str = "127.0.0.1"
    port: int = 9000
    sam3_checkpoint: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            self.host = validate_ipv4(self.host, label="Host listener IP address")
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if isinstance(self.port, bool) or not isinstance(self.port, int) or not 1 <= self.port <= 65535:
            raise ConfigurationError("Host listener port must be an integer between 1 and 65535.")
        self.sam3_checkpoint = str(self.sam3_checkpoint).strip()

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        result.update({"host": self.host, "port": self.port, "sam3_checkpoint": self.sam3_checkpoint})
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HostListenerConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigurationError("Host listener configuration must be an object.")
        known = {"host", "port", "sam3_checkpoint"}
        return cls(
            host=data.get("host", "127.0.0.1"),
            port=data.get("port", 9000),
            sam3_checkpoint=data.get("sam3_checkpoint", ""),
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known}),
        )


@dataclass(slots=True)
class NodeConfig:
    mode: NodeMode
    host: str
    instruction_port: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in ("local", "remote"):
            raise ConfigurationError("Node mode must be 'local' or 'remote'.")
        try:
            self.host = validate_ipv4(self.host, label="Node host")
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        if self.instruction_port is not None and (
            isinstance(self.instruction_port, bool)
            or not isinstance(self.instruction_port, int)
            or not 1 <= self.instruction_port <= 65535
        ):
            raise ConfigurationError("Instruction port must be an integer between 1 and 65535.")
        if self.mode == "remote" and self.instruction_port is None:
            raise ConfigurationError("Remote nodes require an instruction port.")

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        result.update({"mode": self.mode, "host": self.host})
        if self.instruction_port is not None:
            result["instruction_port"] = self.instruction_port
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeConfig":
        if not isinstance(data, dict):
            raise ConfigurationError("Node configuration must be an object.")
        known = {"mode", "host", "instruction_port"}
        return cls(
            mode=data.get("mode", "local"),
            host=data.get("host", "127.0.0.1"),
            instruction_port=data.get("instruction_port"),
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known}),
        )


@dataclass(slots=True)
class ConfiguredService:
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
    def from_dict(cls, data: dict[str, Any]) -> "ConfiguredService":
        if not isinstance(data, dict):
            raise ConfigurationError("Service configuration must be an object.")
        known = {"node", "service_type", "port", "settings"}
        return cls(
            node=str(data.get("node", "local")),
            spec=ServiceSpec.from_dict(data),
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known}),
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
        if isinstance(self.sam_step, bool) or self.sam_step < 1:
            raise ConfigurationError("sam_step must be at least 1.")
        if self.sam_resize is not None and self.sam_resize < 1:
            raise ConfigurationError("sam_resize must be positive or null.")
        if not 0 <= float(self.sam_confidence) <= 1:
            raise ConfigurationError("sam_confidence must be between 0 and 1.")
        self.prompts = [str(item).strip() for item in self.prompts if str(item).strip()]

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        for name in self.__dataclass_fields__:
            if name != "extra":
                result[name] = copy.deepcopy(getattr(self, name))
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PipelineConfig":
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
            raise ConfigurationError("Priority Map output root cannot be empty.")
        if self.preview != "mjpeg":
            raise ConfigurationError("Priority Map preview must be 'mjpeg'.")

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        result.update({"root": str(self.root), "preview": self.preview})
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PriorityMapOutputConfig":
        if data is None:
            return cls()
        if not isinstance(data, dict):
            raise ConfigurationError("Priority Map output configuration must be an object.")
        known = {"root", "preview"}
        return cls(
            root=Path(str(data.get("root", "outputs"))),
            preview=data.get("preview", "mjpeg"),
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known}),
        )


@dataclass(slots=True)
class PriorityMapToolConfig:
    services: dict[str, ConfiguredService] = field(default_factory=dict)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    output: PriorityMapOutputConfig = field(default_factory=PriorityMapOutputConfig)
    visual_llm_mode: str = "same_as_logical"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.visual_llm_mode not in ("same_as_logical", "separate"):
            raise ConfigurationError("visual_llm_mode must be 'same_as_logical' or 'separate'.")

    def to_dict(self) -> dict[str, Any]:
        result = copy.deepcopy(self.extra)
        result.update(
            {
                "services": {name: service.to_dict() for name, service in self.services.items()},
                "pipeline": self.pipeline.to_dict(),
                "output": self.output.to_dict(),
                "visual_llm_mode": self.visual_llm_mode,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PriorityMapToolConfig":
        if not isinstance(data, dict):
            raise ConfigurationError("Priority Map tool configuration must be an object.")
        raw_services = data.get("services", {})
        if not isinstance(raw_services, dict):
            raise ConfigurationError("Priority Map services must be an object.")
        known = {"services", "pipeline", "output", "visual_llm_mode"}
        return cls(
            services={str(name): ConfiguredService.from_dict(value) for name, value in raw_services.items()},
            pipeline=PipelineConfig.from_dict(data.get("pipeline")),
            output=PriorityMapOutputConfig.from_dict(data.get("output")),
            visual_llm_mode=str(data.get("visual_llm_mode", "same_as_logical")),
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known}),
        )


@dataclass(slots=True)
class AppConfig:
    nodes: dict[str, NodeConfig] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=lambda: {"priority-map": PriorityMapToolConfig()})
    host_listener: HostListenerConfig = field(default_factory=HostListenerConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes.setdefault("local", NodeConfig("local", "127.0.0.1"))
        self.tools.setdefault("priority-map", PriorityMapToolConfig())
        if not isinstance(self.tools["priority-map"], PriorityMapToolConfig):
            raise ConfigurationError("Priority Map tool configuration must be an object.")
        for name, configured in self.priority_map.services.items():
            if configured.node not in self.nodes:
                raise ConfigurationError(f"Service {name!r} references unknown node {configured.node!r}.")

    @property
    def priority_map(self) -> PriorityMapToolConfig:
        return self.tools["priority-map"]

    def to_dict(self) -> dict[str, Any]:
        other_tools = copy.deepcopy({name: value for name, value in self.tools.items() if name != "priority-map"})
        other_tools["priority-map"] = self.priority_map.to_dict()
        result = copy.deepcopy(self.extra)
        result.update(
            {
                "nodes": {name: node.to_dict() for name, node in self.nodes.items()},
                "tools": other_tools,
                "host_listener": self.host_listener.to_dict(),
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        if not isinstance(data, dict):
            raise ConfigurationError("Configuration root must be a JSON object.")
        raw_nodes = data.get("nodes", {})
        if not isinstance(raw_nodes, dict):
            raise ConfigurationError("Nodes configuration must be an object.")
        nodes = {str(name): NodeConfig.from_dict(value) for name, value in raw_nodes.items()}
        if "tools" in data:
            raw_tools = data["tools"]
            if not isinstance(raw_tools, dict):
                raise ConfigurationError("Tools configuration must be an object.")
            raw_priority = raw_tools.get("priority-map", {})
            tools = {name: copy.deepcopy(value) for name, value in raw_tools.items() if name != "priority-map"}
            tools["priority-map"] = PriorityMapToolConfig.from_dict(raw_priority)
        else:
            # Legacy flat format migration.
            services = data.get("services", {})
            pipeline = data.get("pipeline", {})
            tools = {
                "priority-map": PriorityMapToolConfig(
                    services={str(name): ConfiguredService.from_dict(value) for name, value in services.items()},
                    pipeline=PipelineConfig.from_dict(pipeline),
                    output=PriorityMapOutputConfig(Path(str(data.get("output_root", "outputs")))),
                )
            }
        known = {"nodes", "tools", "services", "pipeline", "output_root", "host_listener"}
        return cls(
            nodes=nodes,
            tools=tools,
            host_listener=HostListenerConfig.from_dict(data.get("host_listener")),
            extra=copy.deepcopy({key: value for key, value in data.items() if key not in known}),
        )


class ConfigStore:
    def __init__(self, path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parents[1] / "config.json"
        selected_path: str | Path = path or os.environ.get("ARCADIA_CONFIG") or default_path
        self.path = Path(selected_path)
        self.default_path = Path(__file__).resolve().parents[1] / "default_config.json"

    def load(self) -> AppConfig:
        source = self.path if self.path.exists() else self.default_path
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"Could not load configuration from {source}: {exc}") from exc
        return AppConfig.from_dict(data)

    def save(self, config: AppConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass
