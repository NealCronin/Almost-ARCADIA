from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from core.errors import AnalysisError
from core.inference.llm_client import LLMClient
from core.inference.sam_client import SAMClient


@dataclass(slots=True)
class PipelineResult:
    output_directory: str
    result: Any
    output_paths: list[str] = field(default_factory=list)
    frames_processed: int = 0


class _RemoteSceneUnderstanding:
    def __init__(
        self,
        llm_client: LLMClient,
        configured_prompts: list[str] | None = None,
        model: str = "local-model",
        llm_generation: dict[str, float | int] | None = None,
        debug: bool = False,
        **_: Any,
    ) -> None:
        self.llm_client = llm_client
        self.configured_prompts = configured_prompts or []
        self.model = model
        self.debug = debug
        self.llm_generation = llm_generation or {}
        self.vocabulary: dict[str, float] = {}

    def get_labels(self, image: np.ndarray, task: str, recent_graph_context: dict[str, Any] | None = None):
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise AnalysisError("Could not encode a frame for LLM scene understanding.")
        prompt = (
            "Return only JSON with this shape: "
            '{"labels":{"label":{"reasoning":"short reason","score":0,"edges":[]}}}. '
            "For every label, score must be a numeric 0–100 mission-relevance score for the task, "
            "not SAM, detection, or visual confidence. "
            f"Task: {task}. Recent graph context: {json.dumps(recent_graph_context or {})}"
        )
        response = self.llm_client.chat(
            prompt,
            images=[("image/jpeg", encoded.tobytes())],
            model=self.model,
            temperature=float(self.llm_generation["temperature"]) if "temperature" in self.llm_generation else None,
            top_k=int(self.llm_generation["top_k"]) if "top_k" in self.llm_generation else None,
            min_p=float(self.llm_generation["min_p"]) if "min_p" in self.llm_generation else None,
            top_p=float(self.llm_generation["top_p"]) if "top_p" in self.llm_generation else None,
        )
        try:
            payload = json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"LLM scene response was not JSON: {exc}") from exc
        labels = payload.get("labels", payload)
        if not isinstance(labels, dict):
            raise AnalysisError("LLM scene response must contain a labels object.")
        normalized: dict[str, dict[str, Any]] = {}
        edge_intents: list[dict[str, str]] = []
        allowed = {prompt.lower() for prompt in self.configured_prompts}
        for raw_label, raw_info in labels.items():
            label = str(raw_label).strip()
            if not label or not isinstance(raw_info, dict):
                continue
            if allowed and label.lower() not in allowed:
                continue
            try:
                score = float(raw_info.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            normalized[label] = {
                "reasoning": str(raw_info.get("reasoning", "")),
                "score": score,
            }
            for edge in raw_info.get("edges", []) or []:
                if isinstance(edge, dict) and edge.get("text") and (edge.get("to_label") or edge.get("to_node_id")):
                    item = {"source_label": label, "text": str(edge["text"])[:80]}
                    if edge.get("to_label"):
                        item["to_label"] = str(edge["to_label"])
                    if edge.get("to_node_id"):
                        item["to_node_id"] = str(edge["to_node_id"])
                    edge_intents.append(item)
        if not normalized:
            return None
        return _SceneResult(normalized, edge_intents)


@dataclass(slots=True)
class _SceneResult:
    labels: dict[str, dict[str, Any]]
    edge_intents: list[dict[str, str]]


class _RemoteSegment:
    """Priority Map-compatible segmenter using remote SAM and local optical flow."""

    FLOW_SCALE = 0.05

    def __init__(
        self,
        sam_client: SAMClient,
        sam_thresh: float = 0.25,
        sam_resize: tuple[int, int] | None = None,
        **_: Any,
    ) -> None:
        self.sam_client = sam_client
        self.sam_thresh = sam_thresh
        self.sam_resize = sam_resize
        self.segmentations: list[Any] = []
        self.dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray: np.ndarray | None = None
        self.transform_dx = 0.0
        self.transform_dy = 0.0

    def close(self) -> None:
        """Match Priority Map's segmenter lifecycle without a local predictor."""

    def _get_flow_map(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.prev_gray is None:
            raise RuntimeError("Cannot calculate optical flow without a previous frame.")
        current_gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = current_gray_full.shape[:2]
        flow_size = (
            max(1, int(width * self.FLOW_SCALE)),
            max(1, int(height * self.FLOW_SCALE)),
        )
        current_gray = cv2.resize(current_gray_full, flow_size, interpolation=cv2.INTER_AREA)
        previous_gray = cv2.resize(self.prev_gray, flow_size, interpolation=cv2.INTER_AREA)
        initial_flow = np.zeros((*current_gray.shape, 2), dtype=np.float32)
        flow = self.dis.calc(current_gray, previous_gray, initial_flow)
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow_x = np.asarray(flow[..., 0], dtype=np.float32) / self.FLOW_SCALE
        flow_y = np.asarray(flow[..., 1], dtype=np.float32) / self.FLOW_SCALE
        self.transform_dx = float(np.median(flow_x))
        self.transform_dy = float(np.median(flow_y))
        x_coordinates, y_coordinates = np.meshgrid(np.arange(width), np.arange(height))
        return (
            (x_coordinates + flow_x).astype(np.float32),
            (y_coordinates + flow_y).astype(np.float32),
        )

    @staticmethod
    def _remap_centroid(
        centroid: tuple[int, int] | None,
        map_x: np.ndarray,
        map_y: np.ndarray,
    ) -> tuple[int, int] | None:
        if centroid is None:
            return None
        height, width = map_x.shape[:2]  # type: ignore[index]
        x = max(0, min(width - 1, int(round(centroid[0]))))
        y = max(0, min(height - 1, int(round(centroid[1]))))
        new_x = int(round(x - (map_x[y, x] - x)))
        new_y = int(round(y - (map_y[y, x] - y)))
        return (
            max(0, min(width - 1, new_x)),
            max(0, min(height - 1, new_y)),
        )

    def _propagate(self, image: np.ndarray) -> None:
        if self.prev_gray is None:
            return
        map_x, map_y = self._get_flow_map(image)
        height, width = image.shape[:2]  # type: ignore[index]
        for segmentation in self.segmentations:
            mask = np.asarray(segmentation.mask, dtype=np.uint8)
            if mask.shape[:2] != (height, width):
                mask = np.asarray(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST), dtype=np.uint8)
            segmentation.mask = cv2.remap(
                mask,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            segmentation.centroid = self._remap_centroid(segmentation.centroid, map_x, map_y)

    @staticmethod
    def _score_for(label: str, scene_dict: dict[str, Any]) -> float:
        value = scene_dict.get(label, {})
        if isinstance(value, dict):
            try:
                return float(value.get("score", 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def _replace_with_remote_results(self, image: np.ndarray, scene_dict: dict[str, Any]) -> float:
        from priority_map.modules.Segment import Segmentation

        prompts = [str(label).strip() for label in scene_dict if str(label).strip()]
        if not prompts:
            return 0.0
        started = time.perf_counter()
        result = self.sam_client.segment(
            image,
            prompts,
            confidence=self.sam_thresh,
            resize=self.sam_resize,
        )
        elapsed = time.perf_counter() - started
        height, width = image.shape[:2]  # type: ignore[index]
        replacement: list[Any] = []
        for index, raw_mask in enumerate(result.masks):
            mask = np.asarray(raw_mask, dtype=np.uint8)
            if mask.ndim > 2:
                mask = np.squeeze(mask)
            if mask.shape[:2] != (height, width):
                mask = np.asarray(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST), dtype=np.uint8)
            label = result.labels[index] if index < len(result.labels) else prompts[min(index, len(prompts) - 1)]
            if label not in scene_dict:
                label = prompts[min(index, len(prompts) - 1)]
            centroid = None
            if index < len(result.bounding_boxes):
                box = result.bounding_boxes[index]
                if len(box) >= 4:
                    x_scale = width / self.sam_resize[0] if self.sam_resize else 1.0
                    y_scale = height / self.sam_resize[1] if self.sam_resize else 1.0
                    centroid = (
                        max(0, min(width - 1, int(round((float(box[0]) + float(box[2])) * x_scale / 2)))),
                        max(0, min(height - 1, int(round((float(box[1]) + float(box[3])) * y_scale / 2)))),
                    )
            replacement.append(
                Segmentation(
                    mask=mask,
                    label=label,
                    id="",
                    # Priority Map defines this as scene_dict[label]["score"], a semantic priority score,
                    # not SAM confidence.
                    score=self._score_for(label, scene_dict),
                    centroid=centroid,
                    geo_pos=None,
                )
            )
        self.segmentations = replacement
        return elapsed

    def get_segmentations(self, image: np.ndarray, scene_dict: dict[str, Any] | None):
        from priority_map.modules.Segment import SegmentationResult

        self._propagate(image)
        sam3_seconds = self._replace_with_remote_results(image, scene_dict) if scene_dict is not None else 0.0
        self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return SegmentationResult(
            segmentations=self.segmentations,
            sam3_seconds=sam3_seconds,
            flow_transform=(self.transform_dx, self.transform_dy),
        )


class PriorityMapAdapter:
    """The only Almost ARCADIA integration boundary for external Priority Map."""

    PRIORITY_MAP_API_COMMIT = "ea6d1064175b20c1e90dd3f1ffb0b4173f68e03d"

    def __init__(self, runner_factory: Callable[..., Any] | None = None) -> None:
        self.runner_factory = runner_factory

    def run(
        self,
        *,
        input_path: str,
        output_directory: str,
        llm_client: LLMClient,
        sam_client: SAMClient,
        pipeline_settings: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        preview_callback: Callable[[bytes], None] | None = None,
    ) -> PipelineResult:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        settings = dict(pipeline_settings or {})
        input_source = Path(input_path)
        image_folder = self._prepare_input(input_source, output_path, cancel_event=cancel_event)
        if image_folder is None or (cancel_event is not None and cancel_event.is_set()):
            return PipelineResult(
                str(output_path),
                None,
                [str(path) for path in output_path.rglob("*") if path.is_file()],
                0,
            )
        source_fps = self._source_fps(input_source)
        runner = self._make_runner(
            image_folder=image_folder,
            output_directory=output_path,
            llm_client=llm_client,
            sam_client=sam_client,
            settings=settings,
        )
        try:
            result = self._run_runner(
                runner,
                progress_callback,
                cancel_event=cancel_event,
                preview_callback=preview_callback,
                run_at_source_fps=bool(settings.get("run_at_source_fps", False)),
                source_fps=source_fps,
            )
        finally:
            close = getattr(runner, "close", None)
            if callable(close):
                close()
        output_paths = [str(path) for path in output_path.rglob("*") if path.is_file()]
        frames_processed = int(getattr(result, "frames_processed", 0) or 0)
        return PipelineResult(str(output_path), result, output_paths, frames_processed)

    def _make_runner(
        self,
        *,
        image_folder: Path,
        output_directory: Path,
        llm_client: LLMClient,
        sam_client: SAMClient,
        settings: dict[str, Any],
    ):
        if self.runner_factory is not None:
            return self.runner_factory(
                input_path=str(image_folder),
                output_directory=str(output_directory),
                llm_client=llm_client,
                sam_client=sam_client,
                settings=settings,
            )
        try:
            import priority_map.runner as priority_runner
        except ImportError as exc:
            raise AnalysisError(
                "Priority Map is not installed. Install it from https://github.com/josephletobar/priority_map."
            ) from exc

        required_symbols = ("PriorityMapRunner", "SceneUnderstanding", "Segment")
        missing_symbols = [name for name in required_symbols if not hasattr(priority_runner, name)]
        if missing_symbols:
            raise AnalysisError(
                "Priority Map API is incompatible with this adapter "
                f"(tested commit {self.PRIORITY_MAP_API_COMMIT}; missing {', '.join(missing_symbols)})."
            )
        runner_class = priority_runner.PriorityMapRunner
        if not callable(runner_class):
            raise AnalysisError(
                "Priority Map API is incompatible with this adapter "
                f"(tested commit {self.PRIORITY_MAP_API_COMMIT}; PriorityMapRunner is not callable)."
            )
        original_scene = priority_runner.SceneUnderstanding
        original_segment = priority_runner.Segment
        sam_resize = settings.get("sam_resize")
        if isinstance(sam_resize, int):
            sam_resize = (sam_resize, sam_resize)
        try:
            priority_runner.SceneUnderstanding = lambda **kwargs: _RemoteSceneUnderstanding(
                llm_client,
                configured_prompts=settings.get("prompts", []),
                model=settings.get("scene_model") or "local-model",
                llm_generation=settings.get("llm_generation"),
                debug=bool(settings.get("debug", False)),
            )
            priority_runner.Segment = lambda **kwargs: _RemoteSegment(
                sam_client,
                sam_thresh=float(settings.get("sam_confidence", 0.25)),
                sam_resize=sam_resize,
            )
            return runner_class(
                image_folder=image_folder,
                output_dir=output_directory,
                task=settings.get("task", "Find cars"),
                debrief=settings.get("debrief") or None,
                mask=settings.get("prompts") or [],
                sam_step=int(settings.get("sam_step", 5)),
                sam_thresh=float(settings.get("sam_confidence", 0.25)),
                max_image_edge=settings.get("max_image_edge", 640),
                debug=bool(settings.get("debug", False)),
                record=bool(settings.get("record", True)),
                panoramic=bool(settings.get("panoramic", False)),
                graph_agent=bool(settings.get("graph_agent", False)),
                gps_csv=settings.get("gps_csv"),
                camera_intrinsics=settings.get("camera_intrinsics"),
                scene_model=settings.get("scene_model"),
            )
        finally:
            priority_runner.SceneUnderstanding = original_scene
            priority_runner.Segment = original_segment

    @staticmethod
    def _run_runner(
        runner: Any,
        progress_callback: Callable[[dict[str, Any]], None] | None,
        *,
        cancel_event: threading.Event | None,
        preview_callback: Callable[[bytes], None] | None,
        run_at_source_fps: bool,
        source_fps: float | None,
    ):
        if hasattr(runner, "has_next") and hasattr(runner, "run_frame"):
            frame_delay = 1.0 / source_fps if run_at_source_fps and source_fps and source_fps > 0 else 0.0
            while runner.has_next():
                frame_started = time.monotonic()
                frame_result = runner.run_frame()
                if progress_callback:
                    progress_callback(
                        {
                            "frame_index": getattr(frame_result, "frame_index", None),
                            "frames_processed": getattr(runner, "frames_processed", 0),
                            "image_name": getattr(frame_result, "image_name", None),
                        }
                    )
                if preview_callback:
                    output_frame = getattr(frame_result, "output_frame", None)
                    if output_frame is not None:
                        ok, encoded = cv2.imencode(".jpg", output_frame)
                        if ok:
                            preview_callback(encoded.tobytes())
                if cancel_event is not None and cancel_event.is_set():
                    break
                if not getattr(frame_result, "keep_running", True):
                    break
                if frame_delay:
                    time.sleep(max(0.0, frame_delay - (time.monotonic() - frame_started)))
            return runner.result()
        result = runner.run()
        if progress_callback:
            progress_callback({"frames_processed": getattr(result, "frames_processed", 0)})
        return result

    @staticmethod
    def _source_fps(input_path: Path) -> float | None:
        if input_path.is_dir() or input_path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }:
            return None
        capture = cv2.VideoCapture(str(input_path))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        return fps if fps > 0 else None

    @staticmethod
    def _prepare_input(
        input_path: Path, output_path: Path, *, cancel_event: threading.Event | None = None
    ) -> Path | None:
        if cancel_event is not None and cancel_event.is_set():
            return None
        if input_path.is_dir():
            return input_path
        if not input_path.exists():
            raise AnalysisError(f"Analysis input does not exist: {input_path}")
        frames_dir = output_path / "input_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            if cancel_event is not None and cancel_event.is_set():
                return None
            shutil.copy2(input_path, frames_dir / input_path.name)
            return frames_dir
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise AnalysisError(f"Could not open analysis video: {input_path}")
        index = 0
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return None
                ok, frame = capture.read()
                if not ok:
                    break
                if not cv2.imwrite(str(frames_dir / f"frame_{index:06d}.jpg"), frame):
                    raise AnalysisError(f"Could not write extracted frame {index}")
                index += 1
                if cancel_event is not None and cancel_event.is_set():
                    return None
        finally:
            capture.release()
        if index == 0:
            raise AnalysisError(f"Analysis video contains no readable frames: {input_path}")
        return frames_dir
