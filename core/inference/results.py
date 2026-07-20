from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LLMResult:
    text: str
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    masks: list[Any]
    labels: list[str]
    confidences: list[float]
    bounding_boxes: list[Any]
