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
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Optional

from django.http import (
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import render
from django.utils.decorators import method_decorator
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from .utils.model_host.llama_server_helper import LlamaServerHelper, LlamaServerError

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def json_response(data: dict, status: int = 200) -> JsonResponse:
    """Return a JSON JsonResponse with the given *data* dict."""
    return JsonResponse(data, status=status)


# ------------------------------------------------------------------
# Module-level state for the host portal server
# ------------------------------------------------------------------

_host_server: Optional[HTTPServer] = None
_host_server_thread: Optional[threading.Thread] = None
_host_config: dict[str, Any] = {
    "host_ip": "0.0.0.0",
    "port": 8080,
    "model_path": "",
}


class _HostAPIHandler(BaseHTTPRequestHandler):
    """Minimal request handler that delegates to the host API views."""

    # Silence per-request log lines for health-check style requests
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/evaluate-llm":
            self._handle_evaluate_llm()
        elif self.path == "/api/evaluate-sam3":
            self._handle_evaluate_sam3()
        elif self.path == "/api/host/status":
            self._handle_status()
        else:
            self._respond(404, {"error": "Not found"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/host/status":
            self._handle_status()
        else:
            self._respond(404, {"error": "Not found"})

    # -- Handlers --------------------------------------------------

    def _handle_evaluate_llm(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except Exception as exc:
            return self._respond(400, {"error": f"Invalid JSON: {exc}"})

        model_path = _host_config.get("model_path", "")
        if not model_path:
            return self._respond(400, {"error": "No model_path configured"})

        try:
            helper = LlamaServerHelper(
                model_path=model_path,
                host=_host_config.get("host_ip", "0.0.0.0"),
                port=int(_host_config.get("port", 8080)),
            )
            if not helper.start_server():
                return self._respond(503, {"error": "Failed to start llama server"})

            prompt = body.get("prompt", "")
            temperature = body.get("temperature", 0.7)
            n_predict = body.get("n_predict", 256)

            result = helper.evaluate(prompt, temperature=temperature, n_predict=n_predict)
            return self._respond(200, {"content": result})
        except LlamaServerError as exc:
            return self._respond(502, {"error": str(exc)})
        except Exception as exc:
            logger.exception("LLM evaluation error")
            return self._respond(500, {"error": str(exc)})

    def _handle_evaluate_sam3(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except Exception as exc:
            return self._respond(400, {"error": f"Invalid JSON: {exc}"})

        try:
            from .utils.model_host.sam3_server_helper import Sam3ServerHelper  # noqa: PLC0415
        except ImportError:
            return self._respond(501, {"error": "Sam3ServerHelper not available"})

        frame_b64 = body.get("frame_b64")
        points = body.get("points")
        boxes = body.get("boxes")

        frame = None
        if frame_b64:
            try:
                import numpy as np  # noqa: PLC0415

                raw = base64.b64decode(frame_b64)
                frame = np.frombuffer(raw, dtype=np.uint8)
            except Exception as exc:
                return self._respond(400, {"error": f"Invalid frame_b64: {exc}"})

        if frame is None:
            return self._respond(400, {"error": "frame_b64 is required"})

        try:
            helper = Sam3ServerHelper()
            helper.load_model()
            result = helper.evaluate(frame, points=points, boxes=boxes)
            return self._respond(200, result)
        except Exception as exc:
            logger.exception("SAM3 evaluation error")
            return self._respond(500, {"error": str(exc)})

    def _handle_status(self) -> None:
        status: dict[str, Any] = {
            "llama": {},
            "sam3": {},
            "uptime": time.time(),
        }

        # Llama status
        try:
            from .utils.model_host.llama_server_helper import (  # noqa: PLC0415
                LlamaServerHelper,
            )

            helper = LlamaServerHelper()
            status["llama"] = helper.status()
        except Exception as exc:
            status["llama"] = {"error": str(exc)}

        # SAM3 status
        try:
            from .utils.model_host.sam3_server_helper import Sam3ServerHelper  # noqa: PLC0415

            helper = Sam3ServerHelper()
            helper.load_model()
            status["sam3"] = helper.status()
        except Exception:
            status["sam3"] = {"error": "Sam3ServerHelper not available"}

        return self._respond(200, status)

    # -- Response utilities ----------------------------------------

    def _respond(self, status: int, data: dict) -> None:  # noqa: A002
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ------------------------------------------------------------------
# Views
# ------------------------------------------------------------------

def landing_page(request: HttpRequest) -> HttpResponse:  # noqa: ARG001
    """Render the landing / index page."""
    return render(request, "core_orchestrator/index.html")


def host_portal(request: HttpRequest) -> HttpResponse:
    """
    Host portal view.

    GET  – renders the host portal form.
    POST – starts an HTTP server in a daemon thread to serve the Host API.
    """
    global _host_server, _host_server_thread

    if request.method == "GET":
        return render(request, "core_orchestrator/host_portal.html")

    # POST: configure and start the host API server
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return json_response({"error": "Invalid JSON"}, status=400)

    _host_config.update(
        {
            "host_ip": body.get("host_ip", "0.0.0.0"),
            "port": int(body.get("port", 8080)),
            "model_path": body.get("model_path", ""),
        }
    )

    # Start server in a daemon thread if not already running
    if _host_server is None or _host_server_thread is None or not _host_server_thread.is_alive():
        try:
            host_ip = _host_config["host_ip"]
            port = _host_config["port"]
            _host_server = HTTPServer((host_ip, port), _HostAPIHandler)
            _host_server_thread = threading.Thread(
                target=_host_server.serve_forever,
                daemon=True,
                name="host-api-server",
            )
            _host_server_thread.start()
            logger.info("Host API server started on %s:%d", host_ip, port)
        except Exception as exc:
            logger.exception("Failed to start host API server")
            return json_response({"error": f"Failed to start server: {exc}"}, status=500)

    return json_response(
        {"message": "Host API server started", "host_ip": host_ip, "port": port}
    )


@csrf_exempt
def host_evaluate_llm(request: HttpRequest) -> JsonResponse:
    """
    POST-only view: evaluate a prompt against the local LLM.

    Expects JSON body: {prompt, model_path, temperature, n_predict}
    Returns: {content: ...}
    """
    if request.method != "POST":
        return json_response({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return json_response({"error": "Invalid JSON"}, status=400)

    prompt = body.get("prompt", "")
    model_path = body.get("model_path", _host_config.get("model_path", ""))
    temperature = body.get("temperature", 0.7)
    n_predict = body.get("n_predict", 256)

    if not model_path:
        return json_response({"error": "No model_path configured"}, status=400)

    try:
        helper = LlamaServerHelper(model_path=model_path)
        if not helper.start_server():
            return json_response({"error": "Failed to start llama server"}, status=503)

        result = helper.evaluate(prompt, temperature=temperature, n_predict=n_predict)
        return json_response({"content": result})
    except LlamaServerError as exc:
        return json_response({"error": str(exc)}, status=502)
    except Exception as exc:
        logger.exception("LLM evaluation error")
        return json_response({"error": str(exc)}, status=500)


@csrf_exempt
def host_evaluate_sam3(request: HttpRequest) -> JsonResponse:
    """
    POST-only view: run SAM3 inference.

    Expects JSON body with {frame_b64, points, boxes}.
    Returns: {boxes, masks, scores}
    """
    if request.method != "POST":
        return json_response({"error": "Method not allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return json_response({"error": "Invalid JSON"}, status=400)

    frame_b64 = body.get("frame_b64")
    points = body.get("points")
    boxes = body.get("boxes")

    frame = None
    if frame_b64:
        try:
            import cv2  # noqa: PLC0415

            raw = base64.b64decode(frame_b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            return json_response({"error": f"Invalid frame_b64: {exc}"}, status=400)

    if frame is None:
        return json_response({"error": "frame_b64 is required"}, status=400)

    try:
        from .utils.model_host.sam3_server_helper import Sam3ServerHelper  # noqa: PLC0415

        helper = Sam3ServerHelper()
        helper.load_model()
        result = helper.evaluate(frame, points=points, boxes=boxes)
        return json_response(result)
    except ImportError:
        return json_response({"error": "Sam3ServerHelper not available"}, status=501)
    except Exception as exc:
        logger.exception("SAM3 evaluation error")
        return json_response({"error": str(exc)}, status=500)


@csrf_exempt
def host_status(request: HttpRequest) -> JsonResponse:
    """
    GET view: return status of llama and sam3 services.

    Returns: {llama: {...}, sam3: {...}, uptime: ...}
    """
    if request.method != "GET":
        return json_response({"error": "Method not allowed"}, status=405)

    status: dict[str, Any] = {
        "llama": {},
        "sam3": {},
        "uptime": time.time(),
    }

    # Llama status
    try:
        helper = LlamaServerHelper()
        status["llama"] = helper.status()
    except Exception as exc:
        status["llama"] = {"error": str(exc)}

    # SAM3 status
    try:
        from .utils.model_host.sam3_server_helper import Sam3ServerHelper  # noqa: PLC0415

        helper = Sam3ServerHelper()
        helper.load_model()
        status["sam3"] = helper.status()
    except Exception:
        status["sam3"] = {"error": "Sam3ServerHelper not available"}

    return json_response(status)


def client_portal(request: HttpRequest) -> HttpResponse:  # noqa: ARG001
    """Render the client tool selection page."""
    return render(request, "core_orchestrator/tool_selection.html")


def heatmap_dashboard(request: HttpRequest) -> HttpResponse:  # noqa: ARG001
    """Render the heatmap dashboard with default configuration."""
    default_config = {
        "sam3_mode": "remote",
        "llm_mode": "remote",
        "host_ip": "127.0.0.1",
        "host_port": 8080,
        "model_path": "",
        "weights_path": "",
    }
    return render(request, "core_orchestrator/heatmap_dashboard.html", {"config": default_config})


def heatmap_stream(request: HttpRequest) -> StreamingHttpResponse:
    """
    GET view: return an MJPEG video stream.

    Reads config from request.GET:
        sam3_mode, llm_mode, host_ip, host_port, model_path, weights_path

    Runs a loop: reads a test video/camera, applies local or remote SAM3
    inference, draws bounding boxes, encodes to JPEG, yields as MJPEG frame.
    """
    config = {
        "sam3_mode": request.GET.get("sam3_mode", "remote"),
        "llm_mode": request.GET.get("llm_mode", "remote"),
        "host_ip": request.GET.get("host_ip", "127.0.0.1"),
        "host_port": int(request.GET.get("host_port", 8080)),
        "model_path": request.GET.get("model_path", ""),
        "weights_path": request.GET.get("weights_path", ""),
    }

    def frame_generator():
        """Generate MJPEG frames in a loop."""
        try:
            import cv2  # noqa: PLC0415

            # Open camera (index 0) or test video
            cap = cv2.VideoCapture(0 if config.get("sam3_mode") == "local" else config.get("weights_path", ""))
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Apply SAM3 inference
                if config["sam3_mode"] == "local":
                    frame = _apply_local_sam3(frame, config)
                else:
                    frame = _apply_remote_sam3(frame, config)

                # Encode to JPEG
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpeg_bytes = buf.tobytes()

                # Yield as MJPEG frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg_bytes
                    + b"\r\n"
                )

            cap.release()
        except Exception as exc:
            logger.exception("Stream generation error: %s", exc)

    return StreamingHttpResponse(
        frame_generator(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


def _apply_local_sam3(frame: "ndarray", config: dict) -> "ndarray":  # noqa: F821
    """Apply SAM3 locally and draw bounding boxes."""
    try:
        from .utils.model_host.sam3_server_helper import Sam3ServerHelper  # noqa: PLC0415

        helper = Sam3ServerHelper(checkpoint_path=config.get("weights_path", ""))
        helper.load_model()
        result = helper.evaluate(frame)

        boxes = result.get("boxes", [])
        for box in boxes:
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                x, y, w, h = [int(v) for v in box[:4]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    except Exception as exc:
        logger.warning("Local SAM3 failed: %s", exc)

    return frame


def _apply_remote_sam3(frame: "ndarray", config: dict) -> "ndarray":  # noqa: F821
    """Send frame to remote SAM3 server and draw bounding boxes."""
    try:
        import numpy as np  # noqa: PLC0415
        import requests  # noqa: PLC0415

        # Encode frame to base64
        _, buf = cv2.imencode(".jpg", frame)  # noqa: F821
        frame_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        resp = requests.post(
            f"http://{config['host_ip']}:{config['host_port']}/api/evaluate-sam3",
            json={"frame_b64": frame_b64},
            timeout=10,
        )
        if resp.status_code == 200:
            result = resp.json()
            boxes = result.get("boxes", [])
            for box in boxes:
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    x, y, w, h = [int(v) for v in box[:4]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    except Exception as exc:
        logger.warning("Remote SAM3 failed: %s", exc)

    return frame
