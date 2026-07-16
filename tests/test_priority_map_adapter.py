from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.pipeline.priority_map_adapter import PriorityMapAdapter


@dataclass
class FakeResult:
    frames_processed: int = 2


class FakeRunner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.frames_processed = 0

    def run(self):
        self.frames_processed = 2
        return FakeResult()

    def close(self):
        self.closed = True


def test_adapter_keeps_runner_boundary_and_outputs(tmp_path: Path) -> None:
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return FakeRunner(**kwargs)

    source = tmp_path / "images"
    source.mkdir()
    (source / "frame.jpg").write_bytes(b"image")
    output = tmp_path / "output"
    progress = []
    result = PriorityMapAdapter(factory).run(
        input_path=str(source),
        output_directory=str(output),
        llm_client=object(),
        sam_client=object(),
        pipeline_settings={"sam_step": 5},
        progress_callback=progress.append,
    )
    assert result.output_directory == str(output)
    assert result.result.frames_processed == 2
    assert seen["input_path"] == str(source)
    assert seen["settings"]["sam_step"] == 5
    assert progress == [{"frames_processed": 2}]
