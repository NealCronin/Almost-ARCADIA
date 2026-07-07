"""
views.py

Django views for the core orchestrator application.

Provides landing pages, host-side API endpoints for LLM/SAM3 inference,
and the MJPEG heatmap streaming dashboard.
"""

import base64
import io
import json
import logging
import time
import threading
from typing import Any, Optional, Generator

import cv2
import numpy as np

from django.http import (
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import render
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .utils.model_host.llama_server_helper import LlamaServerHelper, LlamaServerError
from .utils.model_host.sam3_server_helper import Sam3ServerHelper, Sam3Error
from .utils.model_host.remote_client_helper import RemoteClientHelper, RemoteClientError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level state for host configuration
# ------------------------------------------------------------------

_host_config: dict[str, Any] = {
    "host_ip": "0.0.0.0",
    "port": 8080,
    "model_path": "",
    "weights_path": "",
    "running": False,
}

_llama_helper: Optional[LlamaServerHelper] = None
_sam3_helper: Optional[Sam3ServerHelper] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def json_response(data: dict, status: int = 200) -> JsonResponse:
    """Return a JSON JsonResponse with the given data dict."""
    return JsonResponse(data, status=status)


def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a numpy frame to JPEG bytes."""
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


def draw_bounding_boxes(
    frame: np.ndarray,
    boxes: list[list[float]],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on a frame."""
    for box in boxes:
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_heatmap_overlay(
    frame: np.ndarray,
    heat_points: list[tuple[float, float]],
    intensity: float = 0.5,
) -> np.ndarray:
    """Draw heatmap overlay on frame at target coordinates."""
    overlay = frame.copy()
    for x, y in heat_points:
        x, y = int(x), int(y)
        # Draw circular heatmap marker
        cv2.circle(overlay, (x, y), 15, (255, 0, 0), -1)
        cv2.circle(overlay, (x, y), 20, (255, 255, 0), 2)
    
    # Blend overlay
    return cv2.addWeighted(overlay, intensity, frame, 1 - intensity, 0)


# ------------------------------------------------------------------
# Views
# ------------------------------------------------------------------

def landing_page(request: HttpRequest) -> HttpResponse:
    """Render the landing / index page with role selection."""
    return render(request, "core_orchestrator/index.html")


def host_portal(request: HttpRequest) -> HttpResponse:
    """
    Host portal view.

    GET  – renders the host portal form.
    POST – updates host configuration.
    """
    if request.method == "GET":
        return render(request, "core_orchestrator/host_portal.html", {
            "config": _host_config,
        })

    # POST: update host configuration
    if request.content_type != "application/json":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return json_response({"error": "Invalid JSON"}, status=400)
    else:
        body = request.POST.dict()

    _host_config.update({
        "host_ip": body.get("host_ip", _host_config.get("host_ip", "0.0.0.0")),
        "port": int(body.get("port", _host_config.get("port", 8080))),
        "model_path": body.get("model_path", _host_config.get("model_path", "")),
        "weights_path": body.get("weights_path", _host_config.get("weights_path", "")),
        "running": True,
    })

    logger.info("Host configuration updated: %s", _host_config)
    return json_response({
        "message": "Host configuration saved",
        "config": _host_config,
    })


@csrf_exempt
def host_evaluate_llm(request: HttpRequest) -> JsonResponse:
    """
    POST-only view: evaluate a prompt against the local LLM.

    Expects JSON body: {prompt, context, temperature, max_tokens}
    Returns: {content: ...}
    """
    if request.method != "POST":
        return json_response({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return json_response({"error": "Invalid JSON"}, status=400)

    prompt = body.get("prompt", "")
    context = body.get("context", "")
    temperature = float(body.get("temperature", 0.7))
    max_tokens = int(body.get("max_tokens", 512))

    if not prompt:
        return json_response({"error": "prompt is required"}, status=400)

    try:
        # Use global helper or create new one
        global _llama_helper
        if _llama_helper is None:
            model_path = _host_config.get("model_path", body.get("model_path", ""))
            if not model_path:
                return json_response(
                    {"error": "No model_path configured. Please set it in host portal."},
                    status=400,
                )
            _llama_helper = LlamaServerHelper(model_path=model_path)

        # Evaluate with context if provided
        if context:
            result = _llama_helper.evaluate_with_context(prompt, context, temperature=temperature, max_tokens=max_tokens)
        else:
            result = _llama_helper.evaluate(prompt, temperature=temperature, max_tokens=max_tokens)

        return json_response({"content": result})

    except LlamaServerError as exc:
        logger.exception("LLM server error")
        return json_response({"error": str(exc)}, status=502)
    except Exception as exc:
        logger.exception("LLM evaluation error")
        return json_response({"error": str(exc)}, status=500)


@csrf_exempt
def host_evaluate_sam3(request: HttpRequest) -> JsonResponse:
    """
    POST-only view: run SAM3 inference.

    Expects JSON body with {frame_b64, input_points, input_boxes} or file upload.
    Returns: {masks, scores, bbox, target_coords}
    """
    if request.method != "POST":
        return json_response({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return json_response({"error": "Invalid JSON"}, status=400)

    # Handle file upload or base64
    frame = None
    if request.FILES.get("image"):
        # File upload
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            for chunk in request.FILES.get("image").chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        frame = cv2.imread(tmp_path)
        import os
        os.unlink(tmp_path)
    elif body.get("frame_b64"):
        # Base64 encoded frame
        try:
            raw = base64.b64decode(body.get("frame_b64"))
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            return json_response({"error": f"Invalid frame_b64: {exc}"}, status=400)

    if frame is None:
        return json_response({"error": "frame_b64 or image file is required"}, status=400)

    input_points = body.get("input_points")
    input_boxes = body.get("input_boxes")

    try:
        global _sam3_helper
        if _sam3_helper is None:
            weights_path = _host_config.get("weights_path", body.get("weights_path", ""))
            _sam3_helper = Sam3ServerHelper(checkpoint_path=weights_path)

        # Run prediction
        if input_points:
            result = _sam3_helper.predict_from_points(frame, input_points, [1] * len(input_points))
        elif input_boxes:
            result = _sam3_helper.predict_from_box(frame, input_boxes[0] if input_boxes else [0, 0, 0, 0])
        else:
            result = _sam3_helper.predict(frame)

        # Extract target coordinates from masks
        target_coords = _sam3_helper.get_target_coordinates(frame, input_points)

        return json_response({
            "masks": result.get("masks", []),
            "scores": result.get("scores", []),
            "bbox": result.get("bbox", []),
            "target_coords": target_coords,
        })

    except Sam3Error as exc:
        logger.exception("SAM3 error")
        return json_response({"error": str(exc)}, status=502)
    except Exception as exc:
        logger.exception("SAM3 evaluation error")
        return json_response({"error": str(exc)}, status=500)


@csrf_exempt
def host_status(request: HttpRequest) -> JsonResponse:
    """
    GET view: return status of llama and sam3 services.

    Returns: {llama: {...}, sam3: {...}, config: {...}}
    """
    if request.method != "GET":
        return json_response({"error": "Method not allowed"}, status=405)

    status: dict[str, Any] = {
        "llama": {},
        "sam3": {},
        "config": _host_config,
        "uptime": time.time(),
    }

    # Llama status
    try:
        if _llama_helper:
            status["llama"] = _llama_helper.status()
        else:
            status["llama"] = {"status": "not_initialized"}
    except Exception as exc:
        status["llama"] = {"error": str(exc)}

    # SAM3 status
    try:
        if _sam3_helper:
            status["sam3"] = _sam3_helper.status()
        else:
            status["sam3"] = {"status": "not_initialized"}
    except Exception as exc:
        status["sam3"] = {"error": str(exc)}

    return json_response(status)


def client_portal(request: HttpRequest) -> HttpResponse:
    """Render the client tool selection page."""
    return render(request, "core_orchestrator/tool_selection.html")


def heatmap_dashboard(request: HttpRequest) -> HttpResponse:
    """Render the heatmap dashboard with configuration."""
    config = {
        "sam3_mode": request.GET.get("sam3_mode", "remote"),
        "llm_mode": request.GET.get("llm_mode", "remote"),
        "host_ip": request.GET.get("host_ip", _host_config.get("host_ip", "127.0.0.1")),
        "host_port": int(request.GET.get("host_port", _host_config.get("port", 8080))),
        "model_path": request.GET.get("model_path", _host_config.get("model_path", "")),
        "weights_path": request.GET.get("weights_path", _host_config.get("weights_path", "")),
        "dataset_path": request.GET.get("dataset_path", ""),
    }
    return render(request, "core_orchestrator/heatmap_dashboard.html", {"config": config})


def heatmap_stream(request: HttpRequest) -> StreamingHttpResponse:
    """
    GET view: return an MJPEG video stream with heatmap overlay.

    Reads config from request.GET:
        sam3_mode, llm_mode, host_ip, host_port, model_path, weights_path

    Runs a loop: reads frames, applies SAM3 inference, draws heatmap,
    encodes to JPEG, yields as MJPEG frame.
    """
    config = {
        "sam3_mode": request.GET.get("sam3_mode", "remote"),
        "llm_mode": request.GET.get("llm_mode", "remote"),
        "host_ip": request.GET.get("host_ip", "127.0.0.1"),
        "host_port": int(request.GET.get("host_port", 8080)),
        "model_path": request.GET.get("model_path", ""),
        "weights_path": request.GET.get("weights_path", ""),
        "dataset_path": request.GET.get("dataset_path", ""),
    }

    def frame_generator() -> Generator[bytes, None, None]:
        """Generate MJPEG frames in a loop."""
        # Initialize helpers based on mode
        sam3_helper: Optional[Sam3ServerHelper] = None
        remote_client: Optional[RemoteClientHelper] = None

        if config["sam3_mode"] == "local":
            sam3_helper = Sam3ServerHelper(checkpoint_path=config.get("weights_path", ""))
            sam3_helper.initialize()
        else:
            remote_client = RemoteClientHelper(
                base_url=f"http://{config['host_ip']}:{config['host_port']}"
            )

        # Try to open camera or video source
        video_source = config.get("dataset_path", "")
        if not video_source:
            video_source = 0  # Default to camera

        cap = cv2.VideoCapture(video_source if isinstance(video_source, int) else video_source)
        if not cap.isOpened():
            logger.warning("Could not open video source, using test pattern")
            cap = cv2.VideoCapture(0)

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Retry opening camera
                    cap.release()
                    cap = cv2.VideoCapture(0)
                    continue

                # Apply SAM3 inference based on mode
                heat_points: list[tuple[float, float]] = []

                if config["sam3_mode"] == "local":
                    frame, heat_points = _apply_local_sam3(frame, sam3_helper)
                else:
                    frame, heat_points = _apply_remote_sam3(frame, remote_client)

                # Draw heatmap overlay
                if heat_points:
                    frame = draw_heatmap_overlay(frame, heat_points)

                # Draw status text
                cv2.putText(
                    frame,
                    f"Mode: {config['sam3_mode'].upper()} | FPS: {frame_count % 60}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # Encode to JPEG and yield
                jpeg_bytes = encode_frame_to_jpeg(frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )
                frame_count += 1

        except Exception as exc:
            logger.exception("Stream generation error: %s", exc)
        finally:
            cap.release()

    return StreamingHttpResponse(
        frame_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


def _apply_local_sam3(
    frame: np.ndarray,
    helper: Optional[Sam3ServerHelper],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Apply SAM3 locally and draw bounding boxes."""
    heat_points: list[tuple[float, float]] = []

    if helper is None:
        return frame, heat_points

    try:
        result = helper.predict(frame)
        target_coords = helper.get_target_coordinates(frame)

        # Draw bounding boxes
        bbox = result.get("bbox", [])
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            draw_bounding_boxes(frame, [bbox])

        heat_points = target_coords

    except Exception as exc:
        logger.warning("Local SAM3 failed: %s", exc)

    return frame, heat_points


def _apply_remote_sam3(
    frame: np.ndarray,
    client: Optional[RemoteClientHelper],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Send frame to remote SAM3 server and draw bounding boxes."""
    heat_points: list[tuple[float, float]] = []

    if client is None:
        return frame, heat_points

    try:
        # Encode frame to base64
        _, buf = cv2.imencode(".jpg", frame)
        frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        resp = client.evaluate_sam3(frame, input_points=None, input_boxes=None)

        if resp:
            # Extract target coordinates
            target_coords = resp.get("target_coords", [])
            heat_points = [tuple(coord) for coord in target_coords]

            # Draw bounding box if present
            bbox = resp.get("bbox", [])
            if bbox and len(bbox) >= 4:
                draw_bounding_boxes(frame, [bbox])

    except RemoteClientError as exc:
        logger.warning("Remote SAM3 failed: %s", exc)
    except Exception as exc:
        logger.warning("Remote SAM3 error: %s", exc)

    return frame, heat_points
