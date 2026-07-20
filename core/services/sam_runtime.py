from __future__ import annotations

import argparse
import base64
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from core.errors import ServiceStartupError
from core.networking import validate_ipv4
from core.services.sam_checkpoint import SAMCheckpointStore
from core.services.specs import ServiceEndpoint, ServiceSpec

logger = logging.getLogger(__name__)


class SAM3Detection(BaseModel):
    label: str
    confidence: float
    box: list[float] | None = None
    centroid: list[float] | None = None
    mask_png_base64: str | None = None


class SAM3PredictResponse(BaseModel):
    detections: list[SAM3Detection]
    overlay_png_base64: str
    image_width: int
    image_height: int


class SAM3PredictRequest(BaseModel):
    """Canonical request: image_base64, text, and confidence."""

    model_config = ConfigDict(extra="forbid")

    image_base64: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=200)
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "image_base64" not in normalized and "image" in normalized:
            normalized["image_base64"] = normalized["image"]
        if "text" not in normalized:
            prompt = normalized.get("prompt", normalized.get("prompts", ""))
            if isinstance(prompt, list):
                prompt = next((item.strip() for item in prompt if isinstance(item, str) and item.strip()), "")
            normalized["text"] = prompt
        for legacy in ("image", "prompt", "prompts"):
            normalized.pop(legacy, None)
        return normalized

    @field_validator("text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("text must be a string")
        return value.strip()


def _decode_image(encoded: str):
    import cv2
    import numpy as np

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("image must be valid base64") from exc
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("image could not be decoded")
    return image


def resolve_sam_device(requested: str, *, torch_module: Any | None = None) -> str:
    """Resolve and validate the device on the compute host that loads SAM3."""
    normalized = str(requested).strip().lower()
    if normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("SAM3 device must be one of: auto, cuda, mps, cpu.")
    if torch_module is None:
        try:
            import torch as loaded_torch

            torch_module = loaded_torch
        except ImportError as exc:
            raise RuntimeError("SAM3 requires PyTorch to select an inference device.") from exc
    cuda_available = bool(getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)())
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_available = bool(getattr(mps_backend, "is_available", lambda: False)())
    if normalized == "auto":
        return "cuda" if cuda_available else "mps" if mps_available else "cpu"
    if normalized == "cuda" and not cuda_available:
        raise ValueError("SAM3 device 'cuda' was selected, but CUDA is unavailable on this host.")
    if normalized == "mps" and not mps_available:
        raise ValueError("SAM3 device 'mps' was selected, but Apple MPS is unavailable on this host.")
    return normalized


def _load_predictor(checkpoint: Path, confidence: float = 0.25, device: str = "auto") -> Any:
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as exc:
        raise RuntimeError("SAM3 requires the project's Ultralytics/SAM3 installation.") from exc
    if checkpoint.suffix.lower() != ".pt":
        raise ValueError("SAM3 checkpoint must be a .pt file.")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint does not exist: {checkpoint}")
    applied_device = resolve_sam_device(device)
    overrides = {
        "conf": confidence,
        "device": applied_device,
        "task": "segment",
        "mode": "predict",
        "model": str(checkpoint),
        "imgsz": 644,
        "save": False,
        "verbose": False,
    }
    wrapper = _UltralyticsPredictor(SAM3SemanticPredictor(overrides=overrides), applied_device)
    wrapper.initialize()
    return wrapper


class _UltralyticsPredictor:
    def __init__(self, predictor: Any, device: str = "cpu") -> None:
        self.predictor = predictor
        self.device = device
        self.initialized = False

    def initialize(self) -> None:
        if getattr(self.predictor, "model", None) is None:
            self.predictor.setup_model(verbose=False)
        self.initialized = True

    def close(self) -> None:
        reset_image = getattr(self.predictor, "reset_image", None)
        if callable(reset_image):
            reset_image()
        self.initialized = False

    def predict(self, image: Any, text: str, confidence: float) -> list[dict[str, Any]]:
        import numpy as np

        self.initialize()
        if hasattr(self.predictor, "args"):
            self.predictor.args.conf = confidence

        # Ultralytics 8.4 SAM3 semantic inference requires set_image() before
        # calling the predictor with its text keyword.
        try:
            self.predictor.set_image(image)
            results = self.predictor(text=[text])
        finally:
            reset_image = getattr(self.predictor, "reset_image", None)
            if callable(reset_image):
                reset_image()

        if not results:
            return []
        result = results[0]
        mask_data = result.masks.data.cpu().numpy() if result.masks is not None else []
        xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []
        names = getattr(result, "names", {})
        detections: list[dict[str, Any]] = []
        count = max(len(mask_data), len(xyxy), len(scores), len(classes))
        for index in range(count):
            score = float(scores[index]) if index < len(scores) else confidence
            if score < confidence:
                continue
            class_index = int(classes[index]) if index < len(classes) else index
            label = names.get(class_index, text) if isinstance(names, dict) else text
            box = [float(value) for value in np.asarray(xyxy[index]).ravel()[:4]] if index < len(xyxy) else None
            mask = np.asarray(mask_data[index], dtype=np.uint8) if index < len(mask_data) else None
            detections.append({"label": str(label), "confidence": score, "box": box, "mask": mask})
        return detections


class SAMRuntime:
    @staticmethod
    def build_command(spec: ServiceSpec, *, allow_test_command: bool = False) -> list[str]:
        settings = spec.settings
        raw_command = settings.get("command")
        if raw_command is not None:
            if (
                not allow_test_command
                or not isinstance(raw_command, list)
                or not all(isinstance(item, str) for item in raw_command)
            ):
                raise ValueError("command is available only to unit tests.")
            return list(raw_command)
        checkpoint = settings.get("checkpoint")
        if not checkpoint:
            raise ValueError("SAM3 service requires a checkpoint setting.")
        checkpoint_path = SAMCheckpointStore.validate_checkpoint_path(str(checkpoint))
        device = resolve_sam_device(str(settings.get("device", "auto")))
        extra_args = settings.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
            raise ValueError("SAM3 extra_args must be a list of strings.")
        return [
            str(settings.get("python_executable", sys.executable)),
            "-m",
            "core.services.sam_runtime",
            "--host",
            validate_ipv4(str(settings.get("bind_host", "127.0.0.1")), label="SAM3 bind host"),
            "--port",
            str(spec.port),
            "--checkpoint",
            str(checkpoint_path),
            "--confidence",
            str(settings.get("confidence", 0.25)),
            "--device",
            device,
        ] + extra_args

    @staticmethod
    def endpoint(spec: ServiceSpec, public_host: str) -> ServiceEndpoint:
        return ServiceEndpoint(str(spec.settings.get("bind_host", public_host)), spec.port, "sam3")

    @staticmethod
    def readiness_url(endpoint: ServiceEndpoint) -> str:
        return f"{endpoint.base_url}/health"

    @classmethod
    def wait_ready(
        cls,
        process: subprocess.Popen[str],
        endpoint: ServiceEndpoint,
        *,
        timeout: float,
        poll_interval: float = 0.5,
        cancel_event: threading.Event | None = None,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = "service is still loading"
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                raise ServiceStartupError("SAM3 startup cancelled.")
            if process.poll() is not None:
                raise ServiceStartupError(f"SAM3 process exited during startup with code {process.returncode}.")
            try:
                response = requests.get(cls.readiness_url(endpoint), timeout=2)
                if response.ok:
                    return
                last_error = f"readiness returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(poll_interval)
        raise ServiceStartupError(f"SAM3 readiness timed out: {last_error}")

    @classmethod
    def launch(
        cls,
        spec: ServiceSpec,
        *,
        public_host: str,
        log_path: Path,
        allow_test_command: bool = False,
    ) -> tuple[subprocess.Popen[str], IO[str], ServiceEndpoint]:
        command = cls.build_command(spec, allow_test_command=allow_test_command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True, shell=False)
        except Exception:
            log_handle.close()
            raise
        return process, log_handle, cls.endpoint(spec, public_host)


def create_app(
    checkpoint: str | Path,
    predictor: Any | None = None,
    confidence: float = 0.25,
    device: str = "auto",
):
    from fastapi import FastAPI, HTTPException

    if not 0 <= confidence <= 1:
        raise ValueError("SAM3 confidence must be between 0.0 and 1.0.")
    checkpoint_path = Path(checkpoint).expanduser()
    applied_device = resolve_sam_device(device)
    if predictor is None:
        predictor = _load_predictor(checkpoint_path, confidence=confidence, device=applied_device)
    lock = threading.Lock()
    app = FastAPI(title="Almost ARCADIA SAM3 service")

    def encode_mask(mask: Any) -> str | None:
        import cv2
        import numpy as np

        array = np.asarray(mask, dtype=np.uint8).squeeze()
        if array.ndim != 2 or array.size == 0:
            return None
        success, encoded = cv2.imencode(".png", (array > 0).astype(np.uint8) * 255)
        if not success:
            raise ValueError("SAM3 mask could not be encoded.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def centroid_from_mask(mask: Any) -> list[float] | None:
        import numpy as np

        array = np.asarray(mask, dtype=np.uint8).squeeze()
        if array.ndim != 2 or array.size == 0:
            return None
        y_coords, x_coords = np.nonzero(array > 0)
        if not len(x_coords):
            return None
        return [float(np.mean(x_coords)), float(np.mean(y_coords))]

    def render_overlay(image: Any, detections: list[dict[str, Any]]) -> str:
        import cv2
        import numpy as np

        rendered = image.copy()
        colors = ((44, 200, 255), (255, 168, 71), (120, 220, 120), (220, 120, 220))
        height, width = rendered.shape[:2]
        for index, detection in enumerate(detections):
            mask = detection.get("mask")
            if mask is None:
                continue
            array = np.asarray(mask, dtype=np.uint8).squeeze()
            if array.ndim != 2 or array.size == 0:
                continue
            if array.shape != (height, width):
                array = cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
            selected = array > 0
            if not np.any(selected):
                continue
            color: Any = np.asarray(colors[index % len(colors)], dtype=np.float32)
            rendered[selected] = np.clip(rendered[selected] * 0.52 + color * 0.48, 0, 255).astype(np.uint8)
        success, encoded = cv2.imencode(".png", rendered)
        if not success:
            raise ValueError("SAM3 overlay could not be encoded.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            modified = checkpoint_path.stat().st_mtime_ns
        except OSError:
            modified = None
        return {
            "status": "ready",
            "service_type": "sam3",
            "checkpoint": str(checkpoint_path),
            "checkpoint_mtime_ns": modified,
            "device": getattr(predictor, "device", applied_device),
            "initialized": bool(getattr(predictor, "initialized", True)),
        }

    @app.on_event("shutdown")
    def shutdown_predictor() -> None:
        close = getattr(predictor, "close", None)
        if callable(close):
            close()

    @app.post("/v1/predict", response_model=SAM3PredictResponse)
    def predict(request: SAM3PredictRequest) -> SAM3PredictResponse:
        try:
            image = _decode_image(request.image_base64)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            with lock:
                raw_detections = predictor.predict(image, request.text, request.confidence)
            if not isinstance(raw_detections, list):
                raise RuntimeError("SAM predictor returned an invalid result.")
            detections = [
                SAM3Detection(
                    label=str(detection.get("label", request.text)),
                    confidence=float(detection.get("confidence", request.confidence)),
                    box=detection.get("box"),
                    centroid=centroid_from_mask(detection.get("mask")),
                    mask_png_base64=encode_mask(detection.get("mask")),
                )
                for detection in raw_detections
                if isinstance(detection, dict)
            ]
            height, width = image.shape[:2]
            return SAM3PredictResponse(
                detections=detections,
                overlay_png_base64=render_overlay(image, raw_detections),
                image_width=int(width),
                image_height=int(height),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("SAM3 inference failed")
            raise HTTPException(status_code=500, detail=f"SAM3 inference failed: {exc}") from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Almost ARCADIA SAM3 inference service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(
        create_app(args.checkpoint, confidence=args.confidence, device=args.device),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
