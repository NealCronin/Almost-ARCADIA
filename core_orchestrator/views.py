"""
core_orchestrator/views.py

Django view functions for the Drone Target Tracking framework.
"""

import base64
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Optional

import cv2
import numpy as np
from django.http import (
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
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
    "sam3_weights_path": "",  # Host-managed SAM3 weights path
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
    with _log_lock:
        _request_log.append({
            "endpoint": endpoint,
            "details": details,
        })
        # Keep only the last N entries
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

    def _send_json_response(self, data: dict[str, Any], status: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_POST(self) -> None:
        """Handle POST requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json_response({"error": "Invalid JSON"}, status=400)
            return

        endpoint = self.path

        # Route to appropriate handler
        if endpoint == "/api/host/evaluate-llm/":
            self._handle_evaluate_llm(data)
        elif endpoint == "/api/host/evaluate-sam3/":
            self._handle_evaluate_sam3(data)
        else:
            self._send_json_response({"error": "Unknown endpoint"}, status=404)

    def _handle_evaluate_llm(self, data: dict[str, Any]) -> None:
        """Handle LLM evaluation request."""
        try:
            from .utils.model_host.llm_request_helper import LlmRequestHelper

            prompt = data.get("prompt", "")
            model_path = data.get("model_path", "")

            if not prompt:
                self._send_json_response({"error": "prompt is required"}, status=400)
                return

            helper = LlmRequestHelper(model_path=model_path)
            if not helper.initialize():
                self._send_json_response({"error": "Failed to initialize LLM"}, status=503)
                return

            result = helper.generate(prompt)
            log_request("/api/host/evaluate-llm/", {"prompt_length": len(prompt)})
            self._send_json_response(result)

        except Exception as exc:
            logger.exception("LLM evaluation error")
            self._send_json_response({"error": str(exc)}, status=500)

    def _handle_evaluate_sam3(self, data: dict[str, Any]) -> None:
        """Handle SAM3 segmentation request (stateless, uses host config)."""
        try:
            from .utils.model_host.sam3_server_helper import Sam3ServerHelper

            # Decode base64 frame
            frame_b64 = data.get("frame_b64")
            input_points = data.get("input_points")
            input_boxes = data.get("input_boxes")

            if not frame_b64:
                self._send_json_response({"error": "frame_b64 is required"}, status=400)
                return

            # Decode frame
            try:
                raw = base64.b64decode(frame_b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    self._send_json_response({"error": "Failed to decode frame"}, status=400)
                    return
            except Exception as exc:
                self._send_json_response({"error": f"Invalid frame_b64: {exc}"}, status=400)
                return

            # Use host-configured weights path (not from client)
            weights_path = _host_api_config.get("sam3_weights_path", "")
            if not weights_path:
                self._send_json_response(
                    {"error": "SAM3 weights path not configured on Host"},
                    status=400
                )
                return

            # Initialize helper (singleton will cache model in memory)
            helper = Sam3ServerHelper(checkpoint_path=weights_path)
            if not helper.initialize():
                self._send_json_response({"error": "Failed to initialize SAM3"}, status=503)
                return

            # Run prediction
            if input_points:
                result = helper.predict_from_points(frame, input_points, [1] * len(input_points))
            elif input_boxes:
                result = helper.predict_from_box(frame, input_boxes[0])
            else:
                result = helper.predict(frame)

            target_coords = helper.get_target_coordinates(frame, input_points)

            log_request("/api/host/evaluate-sam3/", {"frame_shape": str(frame.shape)})
            self._send_json_response({
                "masks": result.get("masks", []),
                "scores": result.get("scores", []),
                "bbox": result.get("bbox", []),
                "target_coords": target_coords,
            })

        except Exception as exc:
            logger.exception("SAM3 evaluation error")
            self._send_json_response({"error": str(exc)}, status=500)


# ------------------------------------------------------------------
# Django Views
# ------------------------------------------------------------------

def landing_page(request: HttpRequest) -> HttpResponse:
    """Landing page: Route to Host or Client portal."""
    return render(request, "core_orchestrator/index.html")


def host_portal(request: HttpRequest) -> HttpResponse:
    """
    Host Portal: Configure listener IP/Port and SAM3 weights path.

    GET: Render configuration form.
    POST: Start background API listener on specified IP:Port with weights path.
    """
    global _host_api_running, _host_api_thread

    if request.method == "GET":
        return render(request, "core_orchestrator/host_portal.html", {
            "listen_ip": _host_api_config.get("listen_ip", "0.0.0.0"),
            "listen_port": _host_api_config.get("listen_port", 8080),
            "sam3_weights_path": _host_api_config.get("sam3_weights_path", ""),
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
    sam3_weights_path = body.get("sam3_weights_path", "")

    try:
        _host_api_config["listen_ip"] = listen_ip
        _host_api_config["listen_port"] = listen_port
        _host_api_config["sam3_weights_path"] = sam3_weights_path

        # Start HTTP server in daemon thread
        server = HTTPServer((listen_ip, listen_port), HostAPIHandler)
        _host_api_thread = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="host-api-server",
        )
        _host_api_thread.start()
        _host_api_running = True

        logger.info("Host API listener started on %s:%d with SAM3 weights: %s",
                    listen_ip, listen_port, sam3_weights_path)
        return JsonResponse({
            "message": "Host API listener started",
            "listen_ip": listen_ip,
            "listen_port": listen_port,
            "sam3_weights_path": sam3_weights_path,
        })

    except Exception as exc:
        logger.exception("Failed to start Host API listener")
        return JsonResponse({"error": f"Failed to start listener: {exc}"}, status=500)


@csrf_exempt
def host_evaluate_llm(request: HttpRequest) -> JsonResponse:
    """
    Host API Endpoint: Run LLM evaluation (stateless, direct Django view).

    Used when Host is running as part of main Django process.
    Expects: {prompt, model_path}
    Returns: {generated_text, tokens_used}
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    prompt = body.get("prompt")
    model_path = body.get("model_path", "")

    if not prompt:
        return JsonResponse({"error": "prompt is required"}, status=400)
    if not model_path:
        return JsonResponse({"error": "model_path is required"}, status=400)

    try:
        # Lazy import
        from .utils.model_host.llm_request_helper import LlmRequestHelper

        helper = LlmRequestHelper(model_path=model_path)
        if not helper.initialize():
            return JsonResponse({"error": "Failed to initialize LLM"}, status=503)

        result = helper.generate(prompt)
        log_request("/api/host/evaluate-llm/", {"prompt_length": len(prompt)})
        return JsonResponse(result)

    except Exception as exc:
        logger.exception("LLM evaluation error")
        return JsonResponse({"error": str(exc)}, status=500)


@csrf_exempt
def host_evaluate_sam3(request: HttpRequest) -> JsonResponse:
    """
    Host API Endpoint: Run SAM3 segmentation (stateless, direct Django view).

    Used when Host is running as part of main Django process.
    Expects: {frame_b64, input_points, input_boxes}
    Returns: {masks, scores, bbox, target_coords}

    NOTE: weights_path is NO LONGER sent by client.
    The Host reads sam3_weights_path from its own configuration.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    frame_b64 = body.get("frame_b64")
    input_points = body.get("input_points")
    input_boxes = body.get("input_boxes")

    if not frame_b64:
        return JsonResponse({"error": "frame_b64 is required"}, status=400)

    # Decode frame
    try:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JsonResponse({"error": "Failed to decode frame"}, status=400)
    except Exception as exc:
        return JsonResponse({"error": f"Invalid frame_b64: {exc}"}, status=400)

    # Use host-configured weights path (NOT from client payload)
    weights_path = _host_api_config.get("sam3_weights_path", "")
    if not weights_path:
        return JsonResponse(
            {"error": "SAM3 weights path not configured on Host. Please set sam3_weights_path in Host Portal."},
            status=400
        )

    try:
        # Lazy import
        from .utils.model_host.sam3_server_helper import Sam3ServerHelper

        # Initialize helper (singleton will cache model in memory)
        helper = Sam3ServerHelper(checkpoint_path=weights_path)
        if not helper.initialize():
            return JsonResponse({"error": "Failed to initialize SAM3"}, status=503)

        # Run prediction
        if input_points:
            result = helper.predict_from_points(frame, input_points, [1] * len(input_points))
        elif input_boxes:
            result = helper.predict_from_box(frame, input_boxes[0])
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
    return JsonResponse({
        "status": "running" if _host_api_running else "stopped",
        "listen_ip": _host_api_config.get("listen_ip", "0.0.0.0"),
        "listen_port": _host_api_config.get("listen_port", 8080),
        "sam3_weights_path": _host_api_config.get("sam3_weights_path", ""),
    })


def client_portal(request: HttpRequest) -> HttpResponse:
    """Client Portal: Tool selection page."""
    return render(request, "core_orchestrator/tool_selection.html")


def heatmap_dashboard(request: HttpRequest) -> HttpResponse:
    """
    Heatmap Dashboard: Main client interface for drone tracking.

    Reads configuration from session or uses defaults.
    """
    config = {
        "dataset_path": request.GET.get("dataset_path", ""),
        "llm_model_path": request.GET.get("llm_model_path", ""),
        "sam3_weights_path": request.GET.get("sam3_weights_path", ""),
        "routing_mode": request.GET.get("routing_mode", "local"),
        "remote_host_ip": request.GET.get("remote_host_ip", "127.0.0.1"),
        "remote_host_port": request.GET.get("remote_host_port", "8080"),
    }
    return render(request, "core_orchestrator/heatmap_dashboard.html", {"config": config})


def heatmap_stream(request: HttpRequest) -> StreamingHttpResponse:
    """
    MJPEG stream endpoint for heatmap visualization.

    Processes frames from dataset and overlays detection results.
    Supports both local (Django-hosted SAM3) and remote (external Host API) modes.
    """
    dataset_path = request.GET.get("dataset_path", "")
    routing_mode = request.GET.get("routing_mode", "local")
    remote_host_ip = request.GET.get("remote_host_ip", "127.0.0.1")
    remote_host_port = request.GET.get("remote_host_port", "8080")
    sam3_weights_path = request.GET.get("sam3_weights_path", "")

    def create_error_frame(error_msg: str, width: int = 640, height: int = 480) -> bytes:
        """Create a JPEG frame with error message overlay."""
        try:
            # Create a dark background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add error text (simple white text on dark background)
            # Split long messages into multiple lines
            lines = []
            words = error_msg.split()
            current_line = "ERROR:"
            for word in words:
                if len(current_line) + len(word) + 1 > 50:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = current_line + " " + word if current_line != "ERROR:" else word
            lines.append(current_line)
            
            # Draw text lines
            y_offset = 50
            for i, line in enumerate(lines[:5]):  # Max 5 lines
                y_pos = y_offset + i * 30
                # Simple rectangle background for text
                cv2.putText(frame, line, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 50), 2)
            
            # Encode as JPEG
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buffer.tobytes()
        except Exception:
            # Fallback: return a minimal valid JPEG (red frame)
            fallback = np.full((480, 640, 3), (0, 0, 200), dtype=np.uint8)
            _, buffer = cv2.imencode(".jpg", fallback, [cv2.IMWRITE_JPEG_QUALITY, 50])
            return buffer.tobytes()

    def generate():
        """Generate MJPEG stream frames."""
        # Lazy imports to avoid circular dependencies
        cv2_local = None
        try:
            cv2_local = __import__("cv2")
        except ImportError:
            logger.error("OpenCV not available for video stream")
            error_bytes = create_error_frame("OpenCV not installed")
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + error_bytes + b"\r\n"
            )
            return

        # Initialize video capture
        cap = None
        if dataset_path:
            cap = cv2_local.VideoCapture(dataset_path)
            if not cap.isOpened():
                logger.error("Failed to open video source: %s", dataset_path)
                error_bytes = create_error_frame(f"Cannot open: {dataset_path}")
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + error_bytes + b"\r\n"
                )
                return
        else:
            # Try to open default camera
            cap = cv2_local.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Failed to open default camera")
                error_bytes = create_error_frame("No camera available\nPlease specify dataset path")
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + error_bytes + b"\r\n"
                )
                return

        # Set capture properties
        cap.set(cv2_local.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2_local.CAP_PROP_FRAME_HEIGHT, 480)

        # Initialize SAM3 helper based on routing mode
        sam3_helper = None
        remote_client = None
        
        try:
            if routing_mode == "remote":
                # Use remote Host API
                from .utils.model_host.remote_client_helper import RemoteClientHelper
                remote_client = RemoteClientHelper(
                    base_url=f"http://{remote_host_ip}:{remote_host_port}"
                )
            else:
                # Use local SAM3 model
                from .utils.model_host.sam3_server_helper import Sam3ServerHelper
                if not sam3_weights_path:
                    logger.warning("No SAM3 weights path configured, using mock model")
                sam3_helper = Sam3ServerHelper(checkpoint_path=sam3_weights_path)
                sam3_helper.initialize()

            frame_count = 0
            skip_frames = 2  # Process every 3rd frame for performance

            while True:
                ret, frame = cap.read()
                if not ret:
                    # Video ended or read error
                    if dataset_path:
                        # Restart video for looping
                        cap.set(cv2_local.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # For camera, just continue with last frame or show error
                        error_bytes = create_error_frame("Camera stream ended")
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + error_bytes + b"\r\n"
                        )
                        break

                # Skip frames for performance
                if frame_count % skip_frames != 0:
                    # Encode frame as-is
                    _, buffer = cv2_local.imencode(".jpg", frame, [cv2_local.IMWRITE_JPEG_QUALITY, 85])
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )
                    frame_count += 1
                    continue

                frame_count += 1
                
                # Run SAM3 inference
                try:
                    if routing_mode == "remote":
                        # Send to remote Host
                        try:
                            result = remote_client.evaluate_sam3(frame)
                            masks = result.get("masks", [])
                            target_coords = result.get("target_coords", [])
                            scores = result.get("scores", [])
                        except Exception as client_exc:
                            logger.warning("Remote SAM3 request failed: %s", client_exc)
                            masks = []
                            target_coords = []
                            scores = []
                    else:
                        # Use local model
                        if sam3_helper:
                            result = sam3_helper.predict(frame)
                            masks = result.get("masks", [])
                            target_coords = result.get("target_coords", [])
                            scores = result.get("scores", [])
                        else:
                            masks = []
                            target_coords = []
                            scores = []

                    # Overlay detections on frame
                    if masks and len(masks) > 0:
                        # Draw mask overlays
                        for i, mask in enumerate(masks):
                            if len(mask) > 0 and isinstance(mask[0], list):
                                # Convert mask to numpy for contour detection
                                try:
                                    import numpy as np
                                    mask_arr = np.array(mask, dtype=np.uint8)
                                    
                                    # Create overlay
                                    overlay = frame.copy()
                                    alpha = 0.5
                                    
                                    # Find contours
                                    contours, _ = cv2_local.findContours(
                                        mask_arr, cv2_local.RETR_EXTERNAL, cv2_local.CHAIN_APPROX_SIMPLE
                                    )
                                    
                                    # Draw contours
                                    cv2_local.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                                    
                                    # Fill mask with semi-transparent color
                                    for contour in contours:
                                        if len(contour) > 0:
                                            cv2_local.drawContours(overlay, [contour], -1, (100, 200, 100), -1)
                                    
                                    # Blend overlay with original frame
                                    frame = cv2_local.addWeighted(overlay, 1, frame, 1, 0)
                                    
                                    # Draw centroid if available
                                    if i < len(target_coords):
                                        cx, cy = target_coords[i]
                                        cv2_local.circle(frame, (int(cx), int(cy)), 8, (255, 0, 0), -1)
                                        
                                        # Draw score label
                                        if i < len(scores):
                                            score_text = f"{scores[i]:.2f}"
                                            cv2_local.putText(frame, score_text, (int(cx) - 30, int(cy) - 15),
                                                             cv2_local.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                except Exception as mask_exc:
                                    logger.warning("Failed to process mask: %s", mask_exc)
                                    continue

                    # Add mode indicator
                    mode_text = f"Mode: {routing_mode.upper()}"
                    cv2_local.putText(frame, mode_text, (10, 30),
                                     cv2_local.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Add frame counter
                    frame_text = f"Frame: {frame_count}"
                    cv2_local.putText(frame, frame_text, (10, 60),
                                     cv2_local.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                except Exception as infer_exc:
                    logger.warning("Inference error on frame %d: %s", frame_count, infer_exc)
                    # Continue with unprocessed frame

                # Encode and yield frame
                _, buffer = cv2_local.imencode(".jpg", frame, [cv2_local.IMWRITE_JPEG_QUALITY, 85])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

        except Exception as exc:
            logger.exception("Stream error: %s", exc)
            # Send error frame
            error_bytes = create_error_frame(f"Stream Error: {exc}")
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + error_bytes + b"\r\n"
            )
        finally:
            # Clean up resources
            if cap:
                cap.release()

    return StreamingHttpResponse(
        generate(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )
