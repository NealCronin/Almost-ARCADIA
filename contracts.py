"""Shared data contracts for the Almost ARCADIA prototype.

All classes are plain dataclasses with explicit JSON serialization and
deserialization. No framework dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    repository: Optional[str] = None
    filename: Optional[str] = None
    local_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelSpec:
        return cls(**data)


# ---------------------------------------------------------------------------
# ServiceSpec
# ---------------------------------------------------------------------------

@dataclass
class ServiceSpec:
    service_type: str
    port: int
    model: Optional[ModelSpec] = None
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.model is not None:
            d["model"] = self.model.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServiceSpec:
        d = dict(data)
        if "model" in d and d["model"] is not None:
            d["model"] = ModelSpec.from_dict(d["model"])
        return cls(**d)


# ---------------------------------------------------------------------------
# ServiceEndpoint
# ---------------------------------------------------------------------------

@dataclass
class ServiceEndpoint:
    host: str
    port: int
    service_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ServiceEndpoint:
        return cls(**data)


# ---------------------------------------------------------------------------
# NodeConfig
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    name: str
    host: str
    instruction_port: int
    local: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeConfig:
        return cls(**data)


# ---------------------------------------------------------------------------
# RunningService
# ---------------------------------------------------------------------------

@dataclass
class RunningService:
    spec: ServiceSpec
    endpoint: ServiceEndpoint
    runtime_handle: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["spec"] = self.spec.to_dict()
        d["endpoint"] = self.endpoint.to_dict()
        # runtime_handle is never serialised.
        d.pop("runtime_handle", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunningService:
        d = dict(data)
        d["spec"] = ServiceSpec.from_dict(d["spec"])
        d["endpoint"] = ServiceEndpoint.from_dict(d["endpoint"])
        return cls(**d)


# ---------------------------------------------------------------------------
# LanguageRequest
# ---------------------------------------------------------------------------

@dataclass
class LanguageRequest:
    prompt: str
    images: Optional[list[bytes]] = None
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # images (list[bytes]) cannot be represented in JSON, so we
        # convert to None at the JSON boundary.
        d["images"] = None
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageRequest:
        # images will always be None when coming from JSON.
        d = dict(data)
        if "images" in d and d["images"] is not None:
            d["images"] = None
        return cls(**d)


# ---------------------------------------------------------------------------
# LanguageResponse
# ---------------------------------------------------------------------------

@dataclass
class LanguageResponse:
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageResponse:
        return cls(**data)


# ---------------------------------------------------------------------------
# SegmentationRequest
# ---------------------------------------------------------------------------

@dataclass
class SegmentationRequest:
    image: bytes
    prompt: str | list[str]

    def to_dict(self) -> dict[str, Any]:
        # image (bytes) cannot be represented in JSON.
        return {
            "prompt": self.prompt,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationRequest:
        d = dict(data)
        if "image" not in d:
            d["image"] = b""  # stub
        return cls(**d)


# ---------------------------------------------------------------------------
# SegmentationResult
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult:
    masks: list
    labels: list[str]
    confidences: list[float]
    bounding_boxes: list
    source_width: Optional[int] = None
    source_height: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationResult:
        return cls(**data)


# ---------------------------------------------------------------------------
# AnalysisConfig
# ---------------------------------------------------------------------------

@dataclass
class AnalysisConfig:
    input_path: str
    output_path: str
    scene_service: ServiceSpec
    segmentation_service: ServiceSpec
    scene_node: NodeConfig
    segmentation_node: NodeConfig
    pipeline_settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["scene_service"] = self.scene_service.to_dict()
        d["segmentation_service"] = self.segmentation_service.to_dict()
        d["scene_node"] = self.scene_node.to_dict()
        d["segmentation_node"] = self.segmentation_node.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisConfig:
        d = dict(data)
        d["scene_service"] = ServiceSpec.from_dict(d["scene_service"])
        d["segmentation_service"] = ServiceSpec.from_dict(d["segmentation_service"])
        d["scene_node"] = NodeConfig.from_dict(d["scene_node"])
        d["segmentation_node"] = NodeConfig.from_dict(d["segmentation_node"])
        return cls(**d)


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    output_directory: str
    result_files: list[str]
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        return cls(**data)


# ---------------------------------------------------------------------------
# AnalysisWorkspace
# ---------------------------------------------------------------------------

@dataclass
class AnalysisWorkspace:
    root: Path
    log_path: Path
    config_path: Path
    result_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "log_path": str(self.log_path),
            "config_path": str(self.config_path),
            "result_path": str(self.result_path),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisWorkspace:
        return cls(
            root=Path(data["root"]),
            log_path=Path(data["log_path"]),
            config_path=Path(data["config_path"]),
            result_path=Path(data["result_path"]),
        )


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------

def to_json(obj: Any) -> str:
    """Serialize a contract object (or dict) to a JSON string."""
    if hasattr(obj, "to_json"):
        return obj.to_json()
    return json.dumps(obj)


def from_json(text: str) -> dict[str, Any]:
    """Parse a JSON string into a plain dict (no deserialization)."""
    return json.loads(text)
