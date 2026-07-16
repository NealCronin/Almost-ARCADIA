from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LLMResult:
    text: str
    raw_response: dict[str, Any] | None = None


@dataclass(slots=True)
class SegmentationResult:
    masks: list[Any]
    labels: list[str]
    confidences: list[float]
    bounding_boxes: list[Any]
