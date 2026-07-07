"""
views.py

Django views for the core orchestrator application.

Strict Client-Host Split Architecture:
- Host Portal: ONLY captures IP/Port for listening. NO model paths.
- Client Portal: ALL model paths, dataset paths, and routing decisions happen here.
- Host API: Stateless endpoints that process raw payloads and return JSON responses.
"""

import base64
import glob
import io
import json
import logging
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Generator, Optional

import cv2
import numpy as np

from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Module-level state for Host API server (stateless per-request)
# ------------------------------------------------------------------

_host_api_running = False
_host_api_thread: Optional[threading.Thread] = None
_host_api_config: dict[str, Any] = {
    "listen_ip": "0.0.0.0",
    "listen_port": 8080,
}

# Per-request model helpers (lazy initialization)
_request_llama_helper: Optional[Any] = None
_request_sam3_helper: Optional[Any] = None

# Request log for Host Portal display
_request_log: list[dict[str, Any]] = []
_MAX_LOG_ENTRIES = 100
_log_lock = threading.Lock()


# ------------------------------------------------------------------
# Request Logging Utility
# ------------------------------------------------------------------

def log_request(endpoint: str, details: dict[str, Any]) -> None:
    """Thread-safe request logging for Host Portal display."""
    global _request_log
    entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "endpoint": endpoint,
        "details": details,
    }
    with _log_lock:
        _request_log.append(entry)
        if len(_request_log) > _MAX_LOG_ENTRIES:
            _request_log = _request_log[-_MAX_LOG_ENTRIES:]


def get_request_logs() -> list[dict[str, Any]]:
    """Thread-safe retrieval of request logs."""
    with _log_lock:
        return list(reversed(_request_log))


# ------------------------------------------------------------------
# Host API Request Handler (Background Server)
# ------------------------------------------------------------------

class HostAPIHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for Host API endpoints (stateless)."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        """Helper to send JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        """Handle POST requests to API endpoints."""
        if self.path == "/api/host/evaluate-llm/":
            self._handle_evaluate_llm()
        elif self.path == "/api/host/evaluate-sam3/":
            self._handle_evaluate_sam3()
        elif self.path == "/api/host/status/":
            self._handle_status()
        else:
            self._send_json(404, {"error": "Not found"})

    def do_GET(self) -> None:
        """Handle GET requests (status only)."""
        if self.path == "/api/host/status/":
            self._handle_status()
        else:
            self._send_json(404, {"error": "Not found"})

    def _handle_evaluate_llm(self) -> None:
        """Process LLM evaluation request (stateless)."""
        try:
            # Parse request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            prompt = body.get("prompt", "")
            context = body.get("context", "")
            model_path = body.get("model_path", "")

            if not prompt:
                return self._send_json(400, {"error": "prompt is required"})
            if not model_path:
                return self._send_json(400, {"error": "model_path is required"})

            # Log request
            log_request("/api/host/evaluate-llm/", {"prompt_len": len(prompt)})

            # Lazy import and initialize model (per-request)
            try:
                from .utils.model_host.llama_server_helper import LlamaServerHelper
                helper = LlamaServerHelper(model_path=model_path)
                if not helper.start_server():
                    return self._send_json(503, {"error": "Failed to start LLM server"})

                # Evaluate
                result = helper.evaluate_with_context(prompt, context) if context else helper.evaluate(prompt)
                helper.stop_server()

                return self._send_json(200, {"content": result})
            except Exception as exc:
                logger.exception("LLM evaluation error")
                return self._send_json(500, {"error": str(exc)})

        except json.JSONDecodeError:
            return self._send_json(400, {"error": "Invalid JSON"})
        except Exception as exc:
            logger.exception("LLM request handling error")
            return self._send_json(500, {"error": str(exc)})

    def _handle_evaluate_sam3(self) -> None:
        """Process SAM3 evaluation request (stateless)."""
        try:
            # Parse request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}

            frame_b64 = body.get("frame_b64")
            input_points = body.get("input_points")
            weights_path = body.get("weights_path", "")

            if not frame_b64:
                return self._send_json(400, {"error": "frame_b64 is required"})
            if not weights_path:
                return self._send_json(400, {"error": "weights_path is required"})

            # Decode frame
            try:
                raw = base64.b64decode(frame_b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    return self._send_json(400, {"error": "Failed to decode frame"})
            except Exception as exc:
                return self._send_json(400, {"error": f"Invalid frame_b64: {exc}"})

            # Log request
            log_request("/api/host/evaluate-sam3/", {"frame_shape": str(frame.shape)})

            # Lazy import and initialize model (per-request)
            try:
                from .utils.model_host.sam3_server_helper import Sam3ServerHelper
                helper = Sam3ServerHelper(checkpoint_path=weights_path)
                if not helper.initialize():
                    return self._send_json(503, {"error": "Failed to initialize SAM3"})

                # Run prediction
                if input_points:
                    result = helper.predict_from_points(frame, input_points, [1] * len(input_points))
                else:
                    result = helper.predict(frame)

                # Extract target coordinates
                target_coords = helper.get_target_coordinates(frame, input_points)

                return self._send_json(200, {
                    "masks": result.get("masks", []),
                    "scores": result.get("scores", []),
                    "bbox": result.get("bbox", []),
                    "target_coords": target_coords,
                })
            except Exception as exc:
                logger.exception("SAM3 evaluation error")
                return self._send_json(500, {"error": str(exc)})

        except json.JSONDecodeError:
            return self._send_json(400, {"error": "Invalid JSON"})
        except Exception as exc:
            logger.exception("SAM3 request handling error")
            return self._send_json(500, {"error": str(exc)})

    def _handle_status(self) -> None:
        """Return Host API status (stateless)."""
        self._send_json(200, {
            "status": "running" if _host_api_running else "stopped",
            "listen_ip": _host_api_config.get("listen_ip"),
            "listen_port": _host_api_config.get("listen_port"),
        })


# ------------------------------------------------------------------
# Django Views
# ------------------------------------------------------------------

def landing_page(request: HttpRequest) -> HttpResponse:
    """Landing page: Route to Host or Client portal."""
    return render(request, "core_orchestrator/index.html")


def host_portal(request: HttpRequest) -> HttpResponse:
    """
    Host Portal: Configure listening IP/Port ONLY.

    GET: Render configuration form.
    POST: Start background API listener on specified IP:Port.
    """
    global _host_api_running, _host_api_thread

    if request.method == "GET":
        return render(request, "core_orchestrator/host_portal.html", {
            "listen_ip": _host_api_config.get("listen_ip", "0.0.0.0"),
            "listen_port": _host_api_config.get("listen_port", 8080),
            "running": _host_api_running,
        })

    # POST: Start/stop Host API listener
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    action = body.get("action", "start")

    if action == "stop":
        # Stop the background server
        if _host_api_thread and _host_api_thread.is_alive():
            # Note: In production, use proper server shutdown
            pass
        _host_api_running = False
        _host_api_thread = None
        return JsonResponse({"message": "Host API listener stopped"})

    # Start the background server
    listen_ip = body.get("listen_ip", "0.0.0.0")
    listen_port = int(body.get("listen_port", 8080))

    try:
        _host_api_config["listen_ip"] = listen_ip
        _host_api_config["listen_port"] = listen_port

        # Start HTTP server in daemon thread
        server = HTTPServer((listen_ip, listen_port), HostAPIHandler)
        _host_api_thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="host-api-server",
        )
        _host_api_thread.start()
        _host_api_running = True

        logger.info("Host API listener started on %s:%d", listen_ip, listen_port)
        return JsonResponse({
            "message": "Host API listener started",
            "listen_ip": listen_ip,
            "listen_port": listen_port,
        })

    except Exception as exc:
        logger.exception("Failed to start Host API listener")
        return JsonResponse({"error": f"Failed to start listener: {exc}"}, status=500)


@csrf_exempt
def host_evaluate_llm(request: HttpRequest) -> JsonResponse:
    """
    Host API Endpoint: Evaluate LLM prompt (stateless, direct Django view).

    Used when Host is running as part of main Django process.
    Expects: {prompt, context, model_path}
    Returns: {content}
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    prompt = body.get("prompt", "")
    context = body.get("context", "")
    model_path = body.get("model_path", "")

    if not prompt:
        return JsonResponse({"error": "prompt is required"}, status=400)
    if not model_path:
        return JsonResponse({"error": "model_path is required"}, status=400)

    try:
        # Lazy import
        from .utils.model_host.llama_server_helper import LlamaServerHelper

        helper = LlamaServerHelper(model_path=model_path)
        if not helper.start_server():
            return JsonResponse({"error": "Failed to start LLM server"}, status=503)

        result = helper.evaluate_with_context(prompt, context) if context else helper.evaluate(prompt)
        helper.stop_server()

        log_request("/api/host/evaluate-llm/", {"prompt_len": len(prompt)})
        return JsonResponse({"content": result})

    except Exception as exc:
        logger.exception("LLM evaluation error")
        return JsonResponse({"error": str(exc)}, status=500)


@csrf_exempt
def host_evaluate_sam3(request: HttpRequest) -> JsonResponse:
    """
    Host API Endpoint: Run SAM3 segmentation (stateless, direct Django view).

    Used when Host is running as part of main Django process.
    Expects: {frame_b64, weights_path, input_points}
    Returns: {masks, scores, bbox, target_coords}
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    frame_b64 = body.get("frame_b64")
    weights_path = body.get("weights_path", "")
    input_points = body.get("input_points")

    if not frame_b64:
        return JsonResponse({"error": "frame_b64 is required"}, status=400)
    if not weights_path:
        return JsonResponse({"error": "weights_path is required"}, status=400)

    # Decode frame
    try:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JsonResponse({"error": "Failed to decode frame"}, status=400)
    except Exception as exc:
        return JsonResponse({"error": f"Invalid frame_b64: {exc}"}, status=400)

    try:
        # Lazy import
        from .utils.model_host.sam3_server_helper import Sam3ServerHelper

        helper = Sam3ServerHelper(checkpoint_path=weights_path)
        if not helper.initialize():
            return JsonResponse({"error": "Failed to initialize SAM3"}, status=503)

        # Run prediction
        if input_points:
            result = helper.predict_from_points(frame, input_points, [1] * len(input_points))
        else:
            result = helper.predict(frame)

        target_coords = helper.get_target_coordinates(frame, input_points)

        log_request("/api/host/evaluate-sam3/", {"frame_shape": str(frame.shape)})
        return JsonResponse({
            "masks": result.get("masks", []),
            "scores": result.get("scores", []),
            "bbox": result.get("bbox", []),
            "target_coords": target_coords,
        })

    except Exception as exc:
        logger.exception("SAM3 evaluation error")
        return JsonResponse({"error": str(exc)}, status=500)


@csrf_exempt
def host_status(request: HttpRequest) -> JsonResponse:
    """Host API Endpoint: Return status (stateless)."""
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    return JsonResponse({
        "status": "running" if _host_api_running else "stopped",
        "listen_ip": _host_api_config.get("listen_ip"),
        "listen_port": _host_api_config.get("listen_port"),
    })


def client_portal(request: HttpRequest) -> HttpResponse:
    """Client Portal: Tool selection page."""
    return render(request, "core_orchestrator/tool_selection.html")


def heatmap_dashboard(request: HttpRequest) -> HttpResponse:
    """
    Drone Heatmap Dashboard: Client-side configuration.

    ALL paths and routing decisions happen on Client side:
    - dataset_path: Local video/file path
    - llm_model_path: Local LLM weights
    - sam3_weights_path: Local SAM3 weights
    - routing_mode: "local" or "remote"
    - remote_host_ip/port: Only used if routing_mode == "remote"
    """
    config = {
        "dataset_path": request.GET.get("dataset_path", ""),
        "llm_model_path": request.GET.get("llm_model_path", ""),
        "sam3_weights_path": request.GET.get("sam3_weights_path", ""),
        "routing_mode": request.GET.get("routing_mode", "local"),
        "remote_host_ip": request.GET.get("remote_host_ip", "127.0.0.1"),
        "remote_host_port": int(request.GET.get("remote_host_port", 8080)),
    }
    return render(request, "core_orchestrator/heatmap_dashboard.html", {"config": config})


def heatmap_stream(request: HttpRequest) -> StreamingHttpResponse:
    """
    MJPEG Stream: Process local dataset with configured routing.

    Reads from local dataset_path (video or image directory).
    Routes inference based on routing_mode (local vs remote).
    Draws heatmap overlay on detected targets.

    Includes robust client disconnect detection to prevent
    streaming thread leaks and resource exhaustion.
    """
    # Client-side configuration from request
    dataset_path = request.GET.get("dataset_path", "")
    routing_mode = request.GET.get("routing_mode", "local")
    remote_host_ip = request.GET.get("remote_host_ip", "127.0.0.1")
    remote_host_port = int(request.GET.get("remote_host_port", 8080))
    sam3_weights_path = request.GET.get("sam3_weights_path", "")

    def frame_generator() -> Generator[bytes, None, None]:
        """Generate MJPEG frames from dataset with disconnect detection."""
        # Initialize remote client if needed
        remote_client = None
        if routing_mode == "remote":
            try:
                from .utils.model_host.remote_client_helper import RemoteClientHelper
                remote_client = RemoteClientHelper(
                    base_url=f"http://{remote_host_ip}:{remote_host_port}"
                )
            except Exception as exc:
                logger.error("Failed to initialize remote client: %s", exc)

        # Open dataset source
        cap = None
        frame_list: list[str] = []
        frame_index = 0

        if dataset_path:
            # Check if it's a video file
            if os.path.isfile(dataset_path) and dataset_path.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv", ".webm")
            ):
                cap = cv2.VideoCapture(dataset_path)
            # Check if it's an image directory
            elif os.path.isdir(dataset_path):
                image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]
                for pattern in image_patterns:
                    frame_list.extend(glob.glob(os.path.join(dataset_path, pattern)))
                frame_list.sort()
        else:
            # Fallback to camera (shouldn't happen per rules, but safe fallback)
            cap = cv2.VideoCapture(0)

        try:
            while True:
                # Check for client disconnect BEFORE processing frame
                # Django 3.1+ provides is_disconnected() method
                if hasattr(request, "is_disconnected"):
                    if request.is_disconnected():
                        logger.info("Client disconnected, stopping stream")
                        break
                else:
                    # Fallback: check if streaming attribute is set
                    if hasattr(request, "_stream") and request._stream is None:
                        logger.info("Request stream terminated, stopping stream")
                        break

                # Get next frame
                if cap is not None:
                    ret, frame = cap.read()
                    if not ret:
                        # Rewind video or break
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                elif frame_list:
                    # Load image from directory
                    frame_path = frame_list[frame_index % len(frame_list)]
                    frame = cv2.imread(frame_path)
                    if frame is None:
                        frame_index += 1
                        continue
                    frame_index += 1
                else:
                    # Generate test pattern
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "No dataset provided",
                        (200, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                # Run SAM3 inference based on routing mode
                target_coords: list[tuple[float, float]] = []

                if routing_mode == "local":
                    # Local inference
                    try:
                        from .utils.model_host.sam3_server_helper import Sam3ServerHelper

                        helper = Sam3ServerHelper(checkpoint_path=sam3_weights_path)
                        if helper.initialize():
                            result = helper.predict(frame)
                            target_coords = helper.get_target_coordinates(frame)

                            # Draw bounding box
                            bbox = result.get("bbox", [])
                            if bbox and len(bbox) >= 4:
                                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except Exception as exc:
                        logger.warning("Local SAM3 inference failed: %s", exc)

                else:
                    # Remote inference
                    if remote_client:
                        try:
                            import base64 as b64

                            _, buf = cv2.imencode(".jpg", frame)
                            frame_b64 = b64.b64encode(buf.tobytes()).decode("utf-8")

                            resp = remote_client._make_request(
                                "/api/host/evaluate-sam3/",
                                json_data={
                                    "frame_b64": frame_b64,
                                    "weights_path": sam3_weights_path,
                                },
                            )

                            if resp:
                                target_coords = resp.get("target_coords", [])
                                bbox = resp.get("bbox", [])
                                if bbox and len(bbox) >= 4:
                                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        except Exception as exc:
                            logger.warning("Remote SAM3 inference failed: %s", exc)

                # Draw heatmap overlay on target coordinates
                for x, y in target_coords:
                    cv2.circle(frame, (int(x), int(y)), 15, (255, 0, 0), -1)
                    cv2.circle(frame, (int(x), int(y)), 20, (255, 255, 0), 2)

                # Draw status text
                cv2.putText(
                    frame,
                    f"Mode: {routing_mode.upper()} | Targets: {len(target_coords)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # Encode to JPEG and yield
                try:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    jpeg_bytes = buf.tobytes()

                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + jpeg_bytes
                        + b"\r\n"
                    )
                except (ConnectionResetError, BrokenPipeError, OSError) as exc:
                    # Client disconnected during frame transmission
                    logger.info("Connection broken during frame yield: %s", exc)
                    break
                except Exception as exc:
                    # Other encoding errors
                    logger.error("Frame encoding failed: %s", exc)
                    break

        except (GeneratorExit, StopIteration):
            # Generator explicitly terminated
            logger.info("Stream generator terminated")
            raise

        finally:
            # CRITICAL: Always release OpenCV resources
            if cap is not None:
                cap.release()
                logger.debug("VideoCapture released")

    return StreamingHttpResponse(
        frame_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
