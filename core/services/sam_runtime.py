import argparse
import base64
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Any

import requests

from core.errors import ServiceStartupError
from core.services.specs import ServiceEndpoint, ServiceSpec


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


def _load_predictor(checkpoint: Path, confidence: float = 0.25) -> Any:
    try:
        from ultralytics.models.sam import SAM3SemanticPredictor
    except ImportError as exc:
        raise RuntimeError("SAM3 requires the project's ultralytics/SAM3 installation.") from exc
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM3 checkpoint does not exist: {checkpoint}")
    overrides = {
        "conf": confidence,
        "task": "segment",
        "mode": "predict",
        "model": str(checkpoint),
        "imgsz": 644,
        "save": False,
        "verbose": False,
    }
    return _UltralyticsPredictor(SAM3SemanticPredictor(overrides=overrides))


class _UltralyticsPredictor:
    def __init__(self, predictor: Any) -> None:
        self.predictor = predictor

    def predict(self, image: Any, prompts: list[str], confidence: float) -> dict[str, Any]:
        import numpy as np

        if getattr(self.predictor, "model", None) is None:
            self.predictor.setup_model(verbose=False)
        results = self.predictor(image, text=prompts)
        if not results:
            return {"masks": [], "labels": [], "confidences": [], "bounding_boxes": []}
        result = results[0]
        masks: list[Any] = []
        labels: list[str] = []
        confidences: list[float] = []
        boxes: list[Any] = []
        mask_data = result.masks.data.cpu().numpy() if result.masks is not None else []
        xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
        scores = result.boxes.conf.cpu().numpy() if result.boxes is not None else []
        classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes is not None else []
        names = getattr(result, "names", {})
        for index, class_index in enumerate(classes):
            score = float(scores[index]) if index < len(scores) else confidence
            if score < confidence:
                continue
            labels.append(str(names.get(int(class_index), prompts[0] if prompts else "object")))
            confidences.append(score)
            boxes.append(np.asarray(xyxy[index]).tolist())
            masks.append(np.asarray(mask_data[index], dtype=np.uint8).tolist())
        return {"masks": masks, "labels": labels, "confidences": confidences, "bounding_boxes": boxes}


class SAMRuntime:
    """Launch and serve one real, serialized SAM3 predictor."""

    @staticmethod
    def build_command(spec: ServiceSpec, *, allow_test_command: bool = False) -> list[str]:
        settings = spec.settings
        raw_command = settings.get("command")
        if raw_command is not None:
            if not allow_test_command:
                raise ValueError("command is available only to unit tests.")
            if not isinstance(raw_command, list) or not all(isinstance(item, str) for item in raw_command):
                raise ValueError("command must be a list of strings.")
            return list(raw_command)
        checkpoint = settings.get("checkpoint")
        if not checkpoint:
            raise ValueError("SAM3 service requires a checkpoint setting.")
        checkpoint_path = Path(str(checkpoint)).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM3 checkpoint does not exist: {checkpoint_path}")
        extra_args = settings.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(isinstance(item, str) for item in extra_args):
            raise ValueError("extra_args must be a list of strings.")
        return [
            str(settings.get("python_executable", sys.executable)),
            "-m",
            "core.services.sam_runtime",
            "--host",
            str(settings.get("bind_host", "0.0.0.0")),
            "--port",
            str(spec.port),
            "--checkpoint",
            str(checkpoint_path),
            "--confidence",
            str(settings.get("confidence", 0.25)),
        ] + extra_args

    @staticmethod
    def endpoint(spec: ServiceSpec, public_host: str) -> ServiceEndpoint:
        return ServiceEndpoint(host=public_host, port=spec.port, service_type="sam3")

    @staticmethod
    def readiness_url(endpoint: ServiceEndpoint) -> str:
        return f"{endpoint.base_url}/health"

    @staticmethod
    def probe(endpoint: ServiceEndpoint, timeout: float) -> requests.Response:
        return requests.get(SAMRuntime.readiness_url(endpoint), timeout=timeout)

    @classmethod
    def wait_ready(
        cls, process: subprocess.Popen[str], endpoint: ServiceEndpoint, *, timeout: float, poll_interval: float = 0.5
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = "service is still loading"
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise ServiceStartupError(f"SAM3 process exited during startup with code {process.returncode}.")
            try:
                response = cls.probe(endpoint, timeout=min(2.0, poll_interval + 0.5))
                if response.status_code == 200:
                    return
                last_error = f"readiness returned HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(poll_interval)
        raise ServiceStartupError(f"SAM3 readiness timed out: {last_error}")

    @classmethod
    def launch(
        cls, spec: ServiceSpec, *, public_host: str, log_path: Path, allow_test_command: bool = False
    ) -> tuple[subprocess.Popen[str], IO[str], ServiceEndpoint]:
        command = cls.build_command(spec, allow_test_command=allow_test_command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
        except Exception:
            log_handle.close()
            raise
        return process, log_handle, cls.endpoint(spec, public_host)


def create_app(checkpoint: str | Path, predictor: Any | None = None, confidence: float = 0.25):
    """Create the app; predictor injection is intentionally test-only."""
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("SAM HTTP serving requires fastapi and pydantic.") from exc
    checkpoint_path = Path(checkpoint).expanduser()
    if predictor is None:
        predictor = _load_predictor(checkpoint_path, confidence=confidence)
    lock = threading.Lock()
    app = FastAPI(title="Almost ARCADIA SAM3 service")

    class PredictRequest(BaseModel):
        image: str
        prompts: list[str] = Field(default_factory=list)
        confidence: float = Field(default=0.25, ge=0, le=1)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "service_type": "sam3"}

    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> dict[str, Any]:
        prompts = [prompt.strip() for prompt in request.prompts if prompt.strip()]
        if not prompts:
            raise HTTPException(status_code=422, detail="at least one text prompt is required")
        try:
            image = _decode_image(request.image)
            with lock:
                result = predictor.predict(image, prompts, request.confidence)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="SAM predictor returned an invalid result")
        confidences = result.get("confidences") or result.get("scores") or []
        boxes = result.get("bounding_boxes") or result.get("boxes") or []
        return {
            "masks": result.get("masks") or [],
            "labels": [str(label) for label in (result.get("labels") or [])],
            "confidences": [float(score) for score in confidences],
            "bounding_boxes": boxes,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Almost ARCADIA SAM3 inference service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.checkpoint, confidence=args.confidence), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
