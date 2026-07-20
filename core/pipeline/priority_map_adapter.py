from __future__ import annotations

import json
import shutil
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import cv2
import networkx as nx
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


@dataclass(slots=True)
class _SceneResult:
    labels: dict[str, dict[str, Any]]
    edge_intents: list[dict[str, str]]


class _RemoteSceneUnderstanding:
    def __init__(
        self,
        llm_client: LLMClient,
        configured_prompts: list[str] | None = None,
        model: str | None = None,
        llm_generation: dict[str, float | int] | None = None,
        debug: bool = False,
        **_: Any,
    ) -> None:
        self.llm_client = llm_client
        self.configured_prompts = configured_prompts or []
        self.model = model
        self.llm_generation = llm_generation or {}
        self.debug = debug

    def get_labels(self, image: np.ndarray, task: str, recent_graph_context: dict[str, Any] | None = None):
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise AnalysisError("Could not encode a frame for LLM scene understanding.")
        prompt = (
            "Return only JSON with shape "
            '{"labels":{"label":{"reasoning":"short reason","score":0,"edges":[]}}}. '
            "Scores are numeric 0-100 mission relevance scores, not detector confidence. "
            f"Task: {task}. Recent graph context: {json.dumps(recent_graph_context or {})}"
        )
        response = self.llm_client.chat(
            prompt,
            images=[("image/jpeg", encoded.tobytes())],
            model=self.model,
            temperature=float(self.llm_generation.get("temperature", 0.1)),
            max_tokens=int(self.llm_generation.get("max_tokens", 1024)),
        )
        text = response.text.strip().replace("```json", "").replace("```", "").strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"Visual LLM scene response was not JSON: {exc}") from exc
        labels = payload.get("labels", payload)
        if not isinstance(labels, dict):
            raise AnalysisError("Visual LLM scene response must contain a labels object.")
        allowed = {item.casefold() for item in self.configured_prompts}
        normalized: dict[str, dict[str, Any]] = {}
        edge_intents: list[dict[str, str]] = []
        for raw_label, raw_info in labels.items():
            label = str(raw_label).strip()
            if not label or not isinstance(raw_info, dict) or (allowed and label.casefold() not in allowed):
                continue
            try:
                score = float(raw_info.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            normalized[label] = {"reasoning": str(raw_info.get("reasoning", "")), "score": score}
            for edge in raw_info.get("edges", []) or []:
                if not isinstance(edge, dict) or not edge.get("text"):
                    continue
                item = {"source_label": label, "text": str(edge["text"])[:80]}
                if edge.get("to_label"):
                    item["to_label"] = str(edge["to_label"])
                if edge.get("to_node_id"):
                    item["to_node_id"] = str(edge["to_node_id"])
                if len(item) > 2:
                    edge_intents.append(item)
        return _SceneResult(normalized, edge_intents) if normalized else None


class _RemoteSegment:
    """Priority Map-compatible segmenter using direct SAM3 and local optical flow."""

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
        return None

    def _get_flow_map(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.prev_gray is None:
            raise RuntimeError("Cannot calculate optical flow without a previous frame.")
        current_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = current_full.shape[:2]
        flow_size = (max(1, int(width * self.FLOW_SCALE)), max(1, int(height * self.FLOW_SCALE)))
        current = cv2.resize(current_full, flow_size, interpolation=cv2.INTER_AREA)
        previous = cv2.resize(self.prev_gray, flow_size, interpolation=cv2.INTER_AREA)
        flow = self.dis.calc(current, previous, np.zeros((*current.shape, 2), dtype=np.float32))
        flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow_x = np.asarray(flow[..., 0], dtype=np.float32) / self.FLOW_SCALE
        flow_y = np.asarray(flow[..., 1], dtype=np.float32) / self.FLOW_SCALE
        self.transform_dx = float(np.median(flow_x))
        self.transform_dy = float(np.median(flow_y))
        x: Any
        y: Any
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        return (x + flow_x).astype(np.float32), (y + flow_y).astype(np.float32)

    @staticmethod
    def _remap_centroid(centroid: tuple[int, int] | None, map_x: np.ndarray, map_y: np.ndarray):
        if centroid is None:
            return None
        height, width = map_x.shape[:2]
        x = max(0, min(width - 1, int(round(centroid[0]))))
        y = max(0, min(height - 1, int(round(centroid[1]))))
        new_x = int(round(x - (map_x[y, x] - x)))
        new_y = int(round(y - (map_y[y, x] - y)))
        return max(0, min(width - 1, new_x)), max(0, min(height - 1, new_y))

    def _propagate(self, image: np.ndarray) -> None:
        if self.prev_gray is None:
            return
        map_x, map_y = self._get_flow_map(image)
        height, width = image.shape[:2]
        for segmentation in self.segmentations:
            mask = np.asarray(segmentation.mask, dtype=np.uint8)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
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
        try:
            return float(value.get("score", 0.0)) if isinstance(value, dict) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _replace_with_remote_results(self, image: np.ndarray, scene_dict: dict[str, Any]) -> float:
        from priority_map.modules.Segment import Segmentation

        prompts = [str(label).strip() for label in scene_dict if str(label).strip()]
        if not prompts:
            self.segmentations = []
            return 0.0
        started = time.perf_counter()
        result = self.sam_client.segment(image, prompts, confidence=self.sam_thresh, resize=self.sam_resize)
        elapsed = time.perf_counter() - started
        height, width = image.shape[:2]
        replacement: list[Any] = []
        for index, raw_mask in enumerate(result.masks):
            mask = np.asarray(raw_mask, dtype=np.uint8).squeeze()
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            label = result.labels[index] if index < len(result.labels) else prompts[min(index, len(prompts) - 1)]
            if label not in scene_dict:
                label = prompts[min(index, len(prompts) - 1)]
            centroid = None
            if index < len(result.bounding_boxes) and len(result.bounding_boxes[index]) >= 4:
                box = result.bounding_boxes[index]
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
        sam_seconds = self._replace_with_remote_results(image, scene_dict) if scene_dict is not None else 0.0
        self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return SegmentationResult(
            segmentations=self.segmentations,
            sam3_seconds=sam_seconds,
            flow_transform=(self.transform_dx, self.transform_dy),
        )


class _GraphAgent:
    """Pinned Priority Map GraphAgent contract backed by the Logical LLM."""

    def __init__(
        self,
        graph_builder: Any,
        task_description: str,
        node_growth_threshold: int = 30,
        review_hop_cutoff: int = 1,
        model: str | None = None,
        debug: bool = False,
        *,
        llm_client: LLMClient,
        llm_generation: dict[str, float | int] | None = None,
    ) -> None:
        self.graph_builder = graph_builder
        self.task_description = task_description
        self.node_growth_threshold = node_growth_threshold
        self.review_hop_cutoff = review_hop_cutoff
        self.model = model
        self.debug = debug
        self.llm_client = llm_client
        self.llm_generation = llm_generation or {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arcadia-graph-agent")
        self.future: Future[dict[str, Any]] | None = None

    def _debug(self, message: str) -> None:
        if self.debug:
            print(message)

    def should_run(self) -> bool:
        return self.graph_builder.count_unreviewed_nodes() >= self.node_growth_threshold

    def _get_context(self):
        rows, edges, view = self.graph_builder.get_agent_graph_data()
        all_nodes = {row[0]: row[1] for row in rows}
        unreviewed = {row[0] for row in rows if row[2] == 0}
        if not all_nodes or not unreviewed:
            return None, None, view
        graph = nx.Graph()
        graph.add_nodes_from(all_nodes)
        graph.add_weighted_edges_from(edges)
        eligible = set(unreviewed)
        for node_id in unreviewed:
            eligible.update(nx.single_source_shortest_path_length(graph, node_id, cutoff=self.review_hop_cutoff))
        eligible_nodes = {node_id: all_nodes[node_id] for node_id in sorted(eligible, key=str) if node_id in all_nodes}
        subgraph = graph.subgraph(eligible_nodes).copy()
        mst_edges: list[tuple[Any, Any, float]] = []
        if subgraph.number_of_edges():
            mst = nx.minimum_spanning_tree(subgraph, weight="weight")
            mst_edges = [(source, target, float(data["weight"])) for source, target, data in mst.edges(data=True)]
        return eligible_nodes, mst_edges, view

    def _prepare_run(self):
        nodes, edges, view = self._get_context()
        if nodes is None:
            return None
        graph_json = json.dumps(
            {
                "nodes": [{"id": node_id, "score": round(score)} for node_id, score in sorted(nodes.items())],
                "edges": [
                    {
                        "from": source,
                        "from_score": round(nodes[source]),
                        "to": target,
                        "to_score": round(nodes[target]),
                        "dist": round(weight),
                    }
                    for source, target, weight in edges
                    if source in nodes and target in nodes
                ],
            },
            indent=2,
        )
        prompt = (
            "You are reviewing a mission-priority scene graph. Return only JSON with shape "
            '{"reasoning":"...","updates":[{"node_id":"...","delta":0}]}. '
            "Each delta is a bounded score adjustment from -100 to 100. Do not invent node IDs. "
            f"Mission: {self.task_description}\nGraph:\n{graph_json}"
        )
        return {"prompt": prompt, "node_ids": list(nodes), "view": view}

    def is_running(self) -> bool:
        return self.future is not None and not self.future.done()

    def _call_model(self, prompt: str) -> tuple[dict[str, Any], str]:
        result = self.llm_client.chat(
            prompt,
            model=self.model,
            temperature=float(self.llm_generation.get("temperature", 0.1)),
            max_tokens=int(self.llm_generation.get("max_tokens", 1024)),
        )
        raw = result.text.strip().replace("```json", "").replace("```", "").strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start < 0 or end <= start:
            raise AnalysisError(f"Graph Agent response did not contain JSON: {raw}")
        try:
            payload = json.loads(raw[start:end])
        except json.JSONDecodeError as exc:
            raise AnalysisError(f"Graph Agent response was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise AnalysisError("Graph Agent response must be a JSON object.")
        return payload, raw

    def _run_model(self, prompt: str, node_ids: list[Any], view: str):
        started = time.time()
        response, raw = self._call_model(prompt)
        return {"response": response, "raw": raw, "elapsed": time.time() - started, "node_ids": node_ids, "view": view}

    def start_async_if_ready(self) -> bool:
        if self.is_running() or not self.should_run():
            return False
        run = self._prepare_run()
        if run is None:
            return False
        self.future = self.executor.submit(self._run_model, run["prompt"], run["node_ids"], run["view"])
        return True

    def _handle_result(self, result: dict[str, Any]) -> None:
        response = result["response"]
        changes = []
        for update in response.get("updates", []) or []:
            if not isinstance(update, dict) or not update.get("node_id") or update.get("delta") is None:
                continue
            try:
                delta = max(-100.0, min(100.0, float(update["delta"])))
            except (TypeError, ValueError):
                continue
            changed = self.graph_builder.apply_score_delta(update["node_id"], delta, view=result["view"])
            if changed is not None:
                changes.append(changed)
        self.graph_builder.mark_agent_reviewed(result["node_ids"], view=result["view"])
        self._debug(f"Graph Agent reviewed {len(result['node_ids'])} nodes and applied {len(changes)} updates.")

    def poll_finished(self) -> bool:
        if self.future is None or not self.future.done():
            return False
        future = self.future
        self.future = None
        try:
            result = future.result()
        except Exception as exc:
            raise AnalysisError(f"Graph Agent inference failed: {exc}") from exc
        self._handle_result(result)
        return True

    def update_priorities(self) -> None:
        run = self._prepare_run()
        if run is None:
            return
        self._handle_result(self._run_model(run["prompt"], run["node_ids"], run["view"]))

    def close(self) -> None:
        if self.future is not None:
            try:
                result = self.future.result()
                self.future = None
                self._handle_result(result)
            except Exception as exc:
                self._debug(f"Graph Agent shutdown discarded a failed pending result: {exc}")
                self.future = None
        self.executor.shutdown(wait=True, cancel_futures=False)


class PriorityMapAdapter:
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
        visual_llm_client: LLMClient | None = None,
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
        if image_folder is None:
            return PipelineResult(str(output_path), None, [], 0)
        runner = self._make_runner(
            image_folder=image_folder,
            output_directory=output_path,
            llm_client=llm_client,
            visual_llm_client=visual_llm_client or llm_client,
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
                source_fps=self._source_fps(input_source),
            )
        finally:
            close = getattr(runner, "close", None)
            if callable(close):
                close()
        paths = [str(path) for path in output_path.rglob("*") if path.is_file()]
        return PipelineResult(str(output_path), result, paths, int(getattr(result, "frames_processed", 0) or 0))

    def _make_runner(
        self,
        *,
        image_folder: Path,
        output_directory: Path,
        llm_client: LLMClient,
        visual_llm_client: LLMClient,
        sam_client: SAMClient,
        settings: dict[str, Any],
    ):
        if self.runner_factory is not None:
            return self.runner_factory(
                input_path=str(image_folder),
                output_directory=str(output_directory),
                llm_client=llm_client,
                visual_llm_client=visual_llm_client,
                sam_client=sam_client,
                settings=settings,
            )
        try:
            import priority_map.runner as priority_runner
        except ImportError as exc:
            raise AnalysisError("Priority Map is not installed. Install the pipeline extra.") from exc
        required = ("PriorityMapRunner", "SceneUnderstanding", "Segment", "GraphAgent")
        missing = [name for name in required if not hasattr(priority_runner, name)]
        if missing:
            raise AnalysisError(
                f"Priority Map is incompatible with the adapter tested at {self.PRIORITY_MAP_API_COMMIT}; "
                f"missing {', '.join(missing)}."
            )
        original_scene = priority_runner.SceneUnderstanding
        original_segment = priority_runner.Segment
        original_graph = priority_runner.GraphAgent
        sam_resize = settings.get("sam_resize")
        if isinstance(sam_resize, int):
            sam_resize = (sam_resize, sam_resize)

        def graph_factory(
            graph_builder: Any,
            task_description: str,
            node_growth_threshold: int = 30,
            review_hop_cutoff: int = 1,
            model: str | None = None,
            debug: bool = False,
        ) -> _GraphAgent:
            del model
            return _GraphAgent(
                graph_builder,
                task_description,
                node_growth_threshold=node_growth_threshold,
                review_hop_cutoff=review_hop_cutoff,
                model=None,
                debug=debug,
                llm_client=llm_client,
                llm_generation=settings.get("llm_generation") or {},
            )

        try:
            priority_runner.SceneUnderstanding = lambda **kwargs: _RemoteSceneUnderstanding(
                visual_llm_client,
                configured_prompts=settings.get("prompts", []),
                model=None,
                llm_generation=settings.get("visual_llm_generation") or settings.get("llm_generation") or {},
                debug=bool(settings.get("debug", False)),
            )
            priority_runner.Segment = lambda **kwargs: _RemoteSegment(
                sam_client,
                sam_thresh=float(settings.get("sam_confidence", 0.25)),
                sam_resize=sam_resize,
            )
            priority_runner.GraphAgent = graph_factory
            runner = priority_runner.PriorityMapRunner(
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
            priority_runner.GraphAgent = original_graph
        self._install_safe_close(runner)
        return runner

    @staticmethod
    def _install_safe_close(runner: Any) -> None:
        """Pinned runner closes GraphBuilder before GraphAgent; reverse that order."""
        if not all(
            hasattr(runner, name) for name in ("video_output", "heatmap_video_output", "segmentation", "graph_builder")
        ):
            return

        def safe_close(self: Any) -> None:
            if getattr(self, "_closed", False):
                return
            self._closed = True
            steps = [
                self.video_output.close,
                self.heatmap_video_output.close,
                self.segmentation.close,
            ]
            if getattr(self, "graph_agent", None) is not None:
                steps.append(self.graph_agent.close)
            steps.append(self.graph_builder.close)
            errors: list[Exception] = []
            for close in steps:
                try:
                    close()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise AnalysisError(f"Priority Map cleanup failed: {errors[0]}")

        runner.close = MethodType(safe_close, runner)

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
            delay = 1.0 / source_fps if run_at_source_fps and source_fps and source_fps > 0 else 0.0
            while runner.has_next():
                started = time.monotonic()
                frame_result = runner.run_frame()
                if progress_callback:
                    progress_callback(
                        {
                            "frame_index": getattr(frame_result, "frame_index", None),
                            "frames_processed": getattr(runner, "frames_processed", 0),
                            "image_name": getattr(frame_result, "image_name", None),
                        }
                    )
                if preview_callback and getattr(frame_result, "output_frame", None) is not None:
                    ok, encoded = cv2.imencode(".jpg", frame_result.output_frame)
                    if ok:
                        preview_callback(encoded.tobytes())
                if cancel_event is not None and cancel_event.is_set():
                    break
                if not getattr(frame_result, "keep_running", True):
                    break
                if delay:
                    time.sleep(max(0.0, delay - (time.monotonic() - started)))
            return runner.result()
        return runner.run()

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
    def _prepare_input(input_path: Path, output_path: Path, *, cancel_event: threading.Event | None = None):
        if cancel_event is not None and cancel_event.is_set():
            return None
        if input_path.is_dir():
            return input_path
        if not input_path.exists():
            raise AnalysisError(f"Analysis input does not exist: {input_path}")
        frames_dir = output_path / "input_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
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
        finally:
            capture.release()
        if index == 0:
            raise AnalysisError(f"Analysis video contains no readable frames: {input_path}")
        return frames_dir
