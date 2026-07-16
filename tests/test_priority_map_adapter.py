from __future__ import annotations

import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.errors import AnalysisError
from core.inference.results import LLMResult, SegmentationResult
from core.pipeline.priority_map_adapter import PriorityMapAdapter, _RemoteSceneUnderstanding, _RemoteSegment


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


class FrameRunner(FakeRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.has_frames = True

    def has_next(self) -> bool:
        return self.has_frames

    def run_frame(self):
        self.has_frames = False
        self.frames_processed = 1
        return types.SimpleNamespace(
            frame_index=0,
            image_name="source.jpg",
            output_frame=np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8),
            keep_running=False,
        )

    def result(self):
        return FakeResult(frames_processed=self.frames_processed)


@dataclass
class FakeSegmentation:
    mask: np.ndarray
    label: str
    id: str
    score: float
    centroid: tuple[int, int] | None = None
    geo_pos: tuple[float, float] | None = None


@dataclass
class FakeSegmentationResult:
    segmentations: list[FakeSegmentation]
    sam3_seconds: float
    flow_transform: tuple[float, float]


class FakeSAMClient:
    def __init__(self, results: list[SegmentationResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def segment(self, image, prompts, confidence, *, resize):
        self.calls.append({"image": image, "prompts": prompts, "confidence": confidence, "resize": resize})
        return self.results.pop(0)


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def chat(self, prompt, **_):
        self.prompts.append(prompt)
        return LLMResult(self.response)


def install_segment_types(monkeypatch: pytest.MonkeyPatch) -> None:
    priority_map = types.ModuleType("priority_map")
    vars(priority_map)["__path__"] = []
    modules = types.ModuleType("priority_map.modules")
    vars(modules)["__path__"] = []
    segment_module = types.ModuleType("priority_map.modules.Segment")
    segment_module.Segmentation = FakeSegmentation
    segment_module.SegmentationResult = FakeSegmentationResult
    monkeypatch.setitem(sys.modules, "priority_map", priority_map)
    monkeypatch.setitem(sys.modules, "priority_map.modules", modules)
    monkeypatch.setitem(sys.modules, "priority_map.modules.Segment", segment_module)


def image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_remote_segment_first_sam_frame_replaces_segmentations(monkeypatch: pytest.MonkeyPatch) -> None:
    install_segment_types(monkeypatch)
    remote = _RemoteSegment(
        FakeSAMClient(
            [
                SegmentationResult(
                    masks=[[[0, 1], [0, 0]]],
                    labels=["car"],
                    confidences=[0.9],
                    bounding_boxes=[[0, 0, 2, 2]],
                )
            ]
        )
    )

    result = remote.get_segmentations(image(), {"car": {"score": 80}})

    assert len(result.segmentations) == 1
    assert result.segmentations[0].mask.shape == (4, 4)
    assert result.segmentations[0].centroid == (1, 1)
    assert result.segmentations[0].score == 80
    assert remote.prev_gray is not None
    assert result.flow_transform == (0.0, 0.0)
    assert len(remote.sam_client.calls) == 1


def test_scene_understanding_requests_and_retains_semantic_score() -> None:
    client = FakeLLMClient('{"labels":{"car":{"reasoning":"mission target","score":87,"edges":[]}}}')

    result = _RemoteSceneUnderstanding(client).get_labels(image(), "Find mission targets")

    assert result is not None
    assert result.labels["car"]["score"] == 87.0
    assert "numeric 0–100 mission-relevance score" in client.prompts[0]
    assert "not SAM, detection, or visual confidence" in client.prompts[0]


def test_remote_segment_propagates_non_sam_frame_without_request(monkeypatch: pytest.MonkeyPatch) -> None:
    install_segment_types(monkeypatch)
    client = FakeSAMClient(
        [
            SegmentationResult(
                masks=[[[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                labels=["car"],
                confidences=[0.9],
                bounding_boxes=[[1, 1, 3, 3]],
            )
        ]
    )
    remote = _RemoteSegment(client)
    remote.get_segmentations(image(), {"car": {"score": 80}})

    def flow_map(_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        remote.transform_dx = 1.0
        remote.transform_dy = 0.0
        x, y = np.meshgrid(np.arange(4), np.arange(4))
        return (x + 1).astype(np.float32), y.astype(np.float32)

    monkeypatch.setattr(remote, "_get_flow_map", flow_map)
    result = remote.get_segmentations(image(), None)

    assert len(client.calls) == 1
    assert result.segmentations[0].mask[0, 1] == 1
    assert result.segmentations[0].centroid == (1, 2)
    assert result.flow_transform == (1.0, 0.0)


def test_remote_segment_calculates_displacement_from_dis_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    install_segment_types(monkeypatch)
    remote = _RemoteSegment(FakeSAMClient([]))
    remote.prev_gray = np.zeros((20, 20), dtype=np.uint8)

    class DeterministicDIS:
        def calc(self, current, previous, initial):
            flow = np.zeros((*current.shape, 2), dtype=np.float32)
            flow[..., 0] = 0.1
            flow[..., 1] = -0.05
            return flow

    remote.dis = DeterministicDIS()
    map_x, map_y = remote._get_flow_map(np.zeros((20, 20, 3), dtype=np.uint8))

    assert remote.transform_dx == pytest.approx(2.0)
    assert remote.transform_dy == pytest.approx(-1.0)
    assert map_x[4, 4] == pytest.approx(6.0)
    assert map_y[4, 4] == pytest.approx(3.0)


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


def install_runner_module(monkeypatch: pytest.MonkeyPatch, runner_class: object) -> types.ModuleType:
    priority_map = types.ModuleType("priority_map")
    vars(priority_map)["__path__"] = []
    runner = types.ModuleType("priority_map.runner")
    runner.PriorityMapRunner = runner_class
    runner.SceneUnderstanding = object()
    runner.Segment = object()
    monkeypatch.setitem(sys.modules, "priority_map", priority_map)
    monkeypatch.setitem(sys.modules, "priority_map.runner", runner)
    return runner


def test_adapter_restores_priority_map_symbols_after_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Runner:
        def __init__(self, **kwargs):
            self.scene = sys.modules["priority_map.runner"].SceneUnderstanding
            self.segment = sys.modules["priority_map.runner"].Segment

    runner_module = install_runner_module(monkeypatch, Runner)
    old_scene, old_segment = runner_module.SceneUnderstanding, runner_module.Segment
    adapter = PriorityMapAdapter()
    runner = adapter._make_runner(
        image_folder=tmp_path,
        output_directory=tmp_path,
        llm_client=object(),
        sam_client=object(),
        settings={},
    )
    assert callable(runner.scene)
    assert callable(runner.segment)
    assert runner_module.SceneUnderstanding is old_scene
    assert runner_module.Segment is old_segment


def test_adapter_preview_uses_rendered_runner_frame(tmp_path: Path) -> None:
    source = tmp_path / "images"
    source.mkdir()
    cv2.imwrite(str(source / "source.jpg"), np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8))
    previews: list[bytes] = []

    result = PriorityMapAdapter(FrameRunner).run(
        input_path=str(source),
        output_directory=str(tmp_path / "output"),
        llm_client=object(),
        sam_client=object(),
        preview_callback=previews.append,
    )

    decoded = cv2.imdecode(np.frombuffer(previews[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert result.frames_processed == 1
    assert len(previews) == 1
    assert decoded is not None
    assert decoded[0, 0, 2] > 240
    assert decoded[0, 0, 0] < 15


def test_adapter_cancellation_during_video_preparation_keeps_partial_frames(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    cancelled = threading.Event()
    created: list[Path] = []

    class Capture:
        released = False

        def isOpened(self) -> bool:
            return True

        def read(self):
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    capture = Capture()

    def write_frame(path: str, _frame: np.ndarray) -> bool:
        frame_path = Path(path)
        frame_path.write_bytes(b"frame")
        created.append(frame_path)
        cancelled.set()
        return True

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: capture)
    monkeypatch.setattr(cv2, "imwrite", write_frame)
    factory_called = False

    def factory(**_):
        nonlocal factory_called
        factory_called = True
        return FakeRunner()

    result = PriorityMapAdapter(factory).run(
        input_path=str(source),
        output_directory=str(tmp_path / "output"),
        llm_client=object(),
        sam_client=object(),
        cancel_event=cancelled,
    )

    assert capture.released
    assert not factory_called
    assert result.result is None
    assert result.frames_processed == 0
    assert created == [tmp_path / "output" / "input_frames" / "frame_000000.jpg"]
    assert created[0].is_file()


def test_adapter_restores_priority_map_symbols_after_constructor_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingRunner:
        def __init__(self, **kwargs):
            raise RuntimeError("constructor failure")

    runner_module = install_runner_module(monkeypatch, FailingRunner)
    old_scene, old_segment = runner_module.SceneUnderstanding, runner_module.Segment
    with pytest.raises(RuntimeError, match="constructor failure"):
        PriorityMapAdapter()._make_runner(
            image_folder=tmp_path,
            output_directory=tmp_path,
            llm_client=object(),
            sam_client=object(),
            settings={},
        )
    assert runner_module.SceneUnderstanding is old_scene
    assert runner_module.Segment is old_segment


def test_adapter_reports_incompatible_priority_map_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner_module = install_runner_module(monkeypatch, object())
    del runner_module.Segment
    with pytest.raises(AnalysisError, match="incompatible"):
        PriorityMapAdapter()._make_runner(
            image_folder=tmp_path,
            output_directory=tmp_path,
            llm_client=object(),
            sam_client=object(),
            settings={},
        )
