from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolPresentation:
    stream_content_type: str
    inline_artifact_extensions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    key: str
    display_name: str
    required_services: tuple[str, ...]
    presentation: ToolPresentation


TOOLS = {
    "priority-map": ToolDefinition(
        key="priority-map",
        display_name="Priority Map",
        required_services=("llm", "visual_llm", "sam3"),
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
    )
}
