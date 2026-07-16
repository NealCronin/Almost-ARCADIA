from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.inference.llm_client import LLMClient
from core.inference.sam_client import SAMClient


@dataclass(slots=True)
class PipelineResult:
    output_directory: str
    result: Any


class PriorityMapAdapter:
    """
    Thin bridge to the external priority_map project.

    The runner_factory is injected so this repository does not copy or
    tightly couple itself to priority_map internals before integration.
    """

    def __init__(self, runner_factory: Callable[..., Any]) -> None:
        self.runner_factory = runner_factory

    def run(
        self,
        *,
        input_path: str,
        output_directory: str,
        llm_client: LLMClient,
        sam_client: SAMClient,
        pipeline_settings: dict[str, Any] | None = None,
    ) -> PipelineResult:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)

        runner = self.runner_factory(
            input_path=input_path,
            output_directory=str(output_path),
            llm_client=llm_client,
            sam_client=sam_client,
            settings=pipeline_settings or {},
        )

        try:
            result = runner.run()
        finally:
            close = getattr(runner, "close", None)
            if callable(close):
                close()

        return PipelineResult(
            output_directory=str(output_path),
            result=result,
        )
