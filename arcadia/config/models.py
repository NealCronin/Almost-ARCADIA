"""Application configuration model with JSON serialization."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arcadia.contracts import NodeConfig, ServiceSpec


class ConfigError(Exception):
    """Raised for configuration loading, saving, or deserialization failures."""


@dataclass
class AppConfig:
    """Portable configuration for the Almost ARCADIA client and local compute node.

    Represents desired settings only — no running process handles, active service
    state, PIDs, analysis progress, runtime logs, download progress, or live
    endpoints discovered during execution.
    """

    nodes: dict[str, NodeConfig] = field(default_factory=dict)

    instruction_host: str = "127.0.0.1"
    instruction_port: int = 9000

    scene_service: ServiceSpec | None = None
    segmentation_service: ServiceSpec | None = None

    scene_node: str | None = None
    segmentation_node: str | None = None

    input_path: str = ""
    output_path: str = ""

    pipeline_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON output.

        Delegates nested serialization to each contract's own to_dict().
        Returns a deep copy so the caller can mutate without affecting the source.
        """
        nodes_dict = {name: node.to_dict() for name, node in self.nodes.items()}
        return {
            "nodes": nodes_dict,
            "instruction_host": self.instruction_host,
            "instruction_port": self.instruction_port,
            "scene_service": self.scene_service.to_dict() if self.scene_service is not None else None,
            "segmentation_service": self.segmentation_service.to_dict() if self.segmentation_service is not None else None,
            "scene_node": self.scene_node,
            "segmentation_node": self.segmentation_node,
            "input_path": self.input_path,
            "output_path": self.output_path,
            "pipeline_settings": copy.deepcopy(self.pipeline_settings),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Deserialize from a plain dict.

        Raises ConfigError with a descriptive message if required fields are
        missing or if nested NodeConfig/ServiceSpec deserialization fails.
        """
        if not isinstance(data, dict):
            raise ConfigError(f"Configuration top-level must be a JSON object, got {type(data).__name__}")

        try:
            nodes_raw = data.get("nodes")
            if nodes_raw is None:
                nodes: dict[str, NodeConfig] = {}
            else:
                if not isinstance(nodes_raw, dict):
                    raise ConfigError("Configuration 'nodes' must be a JSON object")
                nodes = {}
                for name, node_data in nodes_raw.items():
                    try:
                        nodes[name] = NodeConfig.from_dict(node_data)
                    except Exception as exc:
                        raise ConfigError(f"Invalid node '{name}': {exc}") from exc

            scene_service_raw = data.get("scene_service")
            scene_service: ServiceSpec | None = None
            if scene_service_raw is not None:
                try:
                    scene_service = ServiceSpec.from_dict(scene_service_raw)
                except Exception as exc:
                    raise ConfigError(f"Invalid scene_service: {exc}") from exc

            segmentation_service_raw = data.get("segmentation_service")
            segmentation_service: ServiceSpec | None = None
            if segmentation_service_raw is not None:
                try:
                    segmentation_service = ServiceSpec.from_dict(segmentation_service_raw)
                except Exception as exc:
                    raise ConfigError(f"Invalid segmentation_service: {exc}") from exc

            return cls(
                nodes=nodes,
                instruction_host=data.get("instruction_host", "127.0.0.1"),
                instruction_port=data.get("instruction_port", 9000),
                scene_service=scene_service,
                segmentation_service=segmentation_service,
                scene_node=data.get("scene_node"),
                segmentation_node=data.get("segmentation_node"),
                input_path=data.get("input_path", ""),
                output_path=data.get("output_path", ""),
                pipeline_settings=copy.deepcopy(data.get("pipeline_settings", {})),
            )
        except ConfigError:
            raise
        except Exception as exc:
            raise ConfigError(f"Failed to deserialize configuration: {exc}") from exc
