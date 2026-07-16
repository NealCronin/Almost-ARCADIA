from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from core.pipeline.priority_map_adapter import PriorityMapAdapter
from web.forms import LLMServiceForm, PipelineForm, SAMServiceForm


@dataclass(frozen=True, slots=True)
class ToolPresentation:
    stream_content_type: str
    inline_artifact_extensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    key: str
    display_name: str
    settings_forms: tuple[type[Any], ...]
    required_services: tuple[str, ...]
    runner_factory: Callable[[], PriorityMapAdapter]
    presentation: ToolPresentation


TOOLS = {
    "priority-map": ToolDefinition(
        key="priority-map",
        display_name="Priority Map",
        settings_forms=(LLMServiceForm, SAMServiceForm, PipelineForm),
        required_services=("llm", "sam3"),
        runner_factory=PriorityMapAdapter,
        presentation=ToolPresentation(
            stream_content_type="multipart/x-mixed-replace",
            inline_artifact_extensions=(
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".webp",
                ".mp4",
                ".webm",
                ".txt",
                ".log",
                ".csv",
                ".json",
            ),
        ),
    ),
}
