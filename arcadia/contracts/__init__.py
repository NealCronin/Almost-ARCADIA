"""Shared data contracts for the Almost ARCADIA prototype.

These classes are plain dataclasses that carry data between modules.
No framework dependencies — only the Python standard library.

JSON-facing contracts (serializable to/from dict/JSON):
    ModelSpec, ServiceSpec, ServiceEndpoint, NodeConfig,
    LanguageResponse, SegmentationResult, AnalysisConfig,
    AnalysisResult, AnalysisWorkspace

Runtime-only contracts (not JSON-serializable — binary or unserializable fields):
    RunningService    — holds an arbitrary runtime_handle
    LanguageRequest   — contains list[bytes] for images
    SegmentationRequest — contains bytes for image data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# JSON-facing contracts
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    """Description of a model to be loaded by a service."""
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


@dataclass
class ServiceSpec:
    """Specification for launching a service."""
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


@dataclass
class ServiceEndpoint:
    """Network endpoint for a running service."""
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


@dataclass
class NodeConfig:
    """Configuration for an inference node."""
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


@dataclass
class LanguageResponse:
    """Response from a language model."""
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageResponse:
        return cls(**data)


@dataclass
class SegmentationResult:
    """Result from an image segmentation model.

    ``masks``: list of per-pixel mask values.
    Each mask element represents a single segmentation mask.
    The exact in-memory representation depends on what the
    segmentation backend produces — typically a list of NumPy arrays
    or 2-D boolean grids, one per detected region. These masks are
    **not** JSON-serializable and are omitted from dict/JSON output.
    Use the masks only within-process; encode them via the future
    SAM API when transmitting over HTTP.
    """
    masks: list
    labels: list[str]
    confidences: list[float]
    bounding_boxes: list
    source_width: Optional[int] = None
    source_height: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # masks may contain non-serializable elements (e.g. numpy arrays).
        # We omit them from the dict; reconstruction requires the
        # original in-memory object.
        d.pop("masks", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationResult:
        d = dict(data)
        d.setdefault("masks", [])
        return cls(**d)


@dataclass
class AnalysisConfig:
    """Configuration for a full analysis pipeline."""
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


@dataclass
class AnalysisResult:
    """Summary of a completed analysis pipeline."""
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


@dataclass
class AnalysisWorkspace:
    """Filesystem workspace for an analysis run."""
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
# Runtime-only contracts (binary / unserializable fields)
# ---------------------------------------------------------------------------

@dataclass
class RunningService:
    """A service that is currently running.

    The ``runtime_handle`` holds an opaque object produced by the
    service launcher (e.g. a subprocess, a connection, or a model
    reference). It may not be serializable and is **never** included
    in any dictionary or JSON representation.
    """

    spec: ServiceSpec
    endpoint: ServiceEndpoint
    runtime_handle: Any = None

    # Public summary — does not include runtime_handle
    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "endpoint": self.endpoint.to_dict(),
        }


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunningService:
        d = dict(data)
        d["spec"] = ServiceSpec.from_dict(d["spec"])
        d["endpoint"] = ServiceEndpoint.from_dict(d["endpoint"])
        return cls(**d)


@dataclass
class LanguageRequest:
    """Request to a language model.

    The ``images`` field carries raw binary image data. This contract
    is intended for in-memory use — the bytes are **not** serialised
    to JSON here. The future inference clients are responsible for
    choosing an encoding (base64, multipart, etc.) when transmitting
    over the network.
    """

    prompt: str
    images: Optional[list[bytes]] = None
    settings: dict[str, Any] = field(default_factory=dict)

    # No to_dict / from_json — this contract is not JSON-facing.

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageRequest:
        """Construct from a dict, but images will be ``None``."""
        d = dict(data)
        d.setdefault("images", None)
        return cls(**d)


@dataclass
class SegmentationRequest:
    """Request to a segmentation model.

    The ``image`` field carries raw binary image data. Like
    ``LanguageRequest``, this contract is for in-memory use only — the
    bytes are **not** serialised to JSON here. The future segmentation
    client is responsible for encoding over the network.
    """

    image: bytes
    prompt: str | list[str]

    # No to_dict / from_json — this contract is not JSON-facing.

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SegmentationRequest:
        """Construct from a dict, but image will be an empty stub."""
        d = dict(data)
        d.setdefault("image", b"")
        return cls(**d)
