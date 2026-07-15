"""SAM 3 segmentation HTTP server.

FastAPI application that exposes /health and /v1/segment endpoints.
The SAM 3 model is loaded lazily on startup.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ultralytics.models.sam import SAM3SemanticPredictor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM 3 segmentation server")
    parser.add_argument("--checkpoint", required=True, help="Path to SAM 3 checkpoint")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, etc.)")
    parser.add_argument(
        "--half-precision",
        default=None,
        type=str,
        choices=["true", "false"],
        help="Use half precision (true/false)",
    )
    parser.add_argument(
        "--default-confidence",
        type=float,
        default=0.25,
        help="Default confidence threshold (0.0-1.0)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

@dataclass
class SAMServerState:
    predictor: Any | None = None
    checkpoint_path: Path | None = None
    half_precision: bool = False
    device: str = "cpu"
    default_confidence: float = 0.25
    ready: bool = False


state = SAMServerState()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()


@app.on_event("startup")
async def startup():
    """Load the SAM 3 model on startup."""
    if state.predictor is not None:
        return
    try:
        predictor = _SAM3Predictor(
            checkpoint=state.checkpoint_path,
            device=state.device,
            half_precision=state.half_precision,
            default_confidence=state.default_confidence,
        )
        state.predictor = predictor
        state.ready = True
        logger.info("SAM 3 model loaded successfully")
    except Exception as e:
        logger.exception("Failed to load SAM 3 model")
        raise


@app.get("/health")
async def health():
    """Health check endpoint."""
    if state.ready:
        return {"status": "ready", "service_type": "segmentation"}
    return {"status": "loading", "service_type": "segmentation"}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SegmentRequestBody(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded image bytes")
    prompts: list[str] = Field(..., min_length=1, description="Text prompts for segmentation")
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence threshold override")


# ---------------------------------------------------------------------------
# Prediction wrapper
# ---------------------------------------------------------------------------

class _SAM3Predictor:
    """Thin wrapper around SAM3SemanticPredictor for the server."""

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        half_precision: bool,
        default_confidence: float,
    ) -> None:
        overrides = {
            "model": str(checkpoint),
            "device": device,
            "half": half_precision,
            "conf": default_confidence,
            "retina_masks": True,
        }
        self._predictor = SAM3SemanticPredictor(overrides=overrides)
        self._predictor.setup_model(model=str(checkpoint))
        self._default_confidence = default_confidence

    def predict(
        self,
        image: np.ndarray,
        prompts: list[str],
        confidence: float,
    ) -> list[dict]:
        """Run segmentation and return predictions.

        Args:
            image: RGB numpy array (H, W, 3)
            prompts: Text prompts for segmentation
            confidence: Confidence threshold

        Returns:
            List of prediction dicts with label, confidence, bounding_box, and mask.
        """
        self._predictor.reset_image()
        self._predictor.reset_prompts()
        self._predictor.set_image(image)
        results = list(self._predictor(text=prompts))
        if not results:
            return []
        result = results[0]
        predictions = []
        if result.masks is not None and result.boxes is not None:
            masks_tensor = result.masks.data
            boxes_tensor = result.boxes.data
            names = result.names
            for i in range(masks_tensor.shape[0]):
                box = boxes_tensor[i].tolist()
                cls_id = int(box[5])
                label = names[cls_id] if isinstance(names, dict) and cls_id in names else str(cls_id)
                predictions.append({
                    "label": label,
                    "confidence": float(box[4]),
                    "bounding_box": [
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ],
                    "mask": masks_tensor[i].cpu().numpy() if hasattr(masks_tensor, 'cpu') else np.array(masks_tensor[i]),
                })
        return predictions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_mask(mask: np.ndarray) -> dict:
    """Convert a 2-D boolean numpy array to a PNG base64 dict."""
    from PIL import Image
    mask_uint8 = mask.astype(np.uint8) * 255
    img = Image.fromarray(mask_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {
        "encoding": "png_base64",
        "data": b64,
        "width": mask.shape[1],
        "height": mask.shape[0],
    }


# ---------------------------------------------------------------------------
# Segment endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/segment")
async def segment(body: SegmentRequestBody):
    """Segment an image using SAM 3."""
    if not state.ready:
        raise HTTPException(status_code=503, detail="Service not ready")

    # Decode image
    try:
        image_bytes = base64.b64decode(body.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    image_array = np.array(img)
    confidence = body.confidence if body.confidence is not None else state.default_confidence

    try:
        predictions = state.predictor.predict(image_array, body.prompts, confidence)
    except Exception as e:
        logger.exception("Segmentation failed")
        raise HTTPException(status_code=500, detail="Segmentation failed")

    response_predictions = []
    for pred in predictions:
        mask_dict = _encode_mask(pred["mask"])
        response_predictions.append({
            "label": pred["label"],
            "confidence": pred["confidence"],
            "bounding_box": pred["bounding_box"],
            "mask": mask_dict,
        })

    return {
        "predictions": response_predictions,
        "source_width": img.width,
        "source_height": img.height,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    _args = parse_args()
    state.checkpoint_path = Path(_args.checkpoint)
    state.half_precision = _args.half_precision == "true"
    state.device = _args.device
    state.default_confidence = _args.default_confidence

    import uvicorn
    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")


if __name__ == "__main__":
    main()