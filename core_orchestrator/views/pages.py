"""
pages.py

Django views for page rendering and API endpoints, refactored to use
the persistent settings store and service manager.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, Optional

import cv2
import numpy as np
from django.http import HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from ..services.command_builder import preview_command
from ..services.service_manager import get_service_manager
from ..services.settings_store import (
    _dataclass_to_dict,
    _dict_to_appsettings,
    get_settings_store,
    AppSettings,
    LLMServiceSettings,
    SAMServiceSettings,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level runtime state (NOT persisted)
# ---------------------------------------------------------------------------
_host_api_running = False
_host_api_thread: Optional[threading.Thread] = None
_host_api_server: Optional[HTTPServer] = None
_host_api_config: dict[str, Any] = {}

# Request log for Host Portal display
_request_log: list[dict[str, Any]] = []
_MAX_LOG_ENTRIES = 100
_log_lock = threading.Lock()
_config_lock = threading.Lock()
_lifecycle_lock = threading.Lock()  # protects start/stop/restart of listener

# Maximum request body size for custom Host listener (50 MB)
_MAX_REQUEST_BODY = 50 * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _error(code: str, message: str, status: int = 400) -> JsonResponse:
    return JsonResponse(
        {"error": {"code": code, "message": message, "details": {}}},
        status=status,
    )


def log_request(endpoint: str, details: dict[str, Any]) -> None:
    """Thread-safe request logging for Host Portal display."""
    with _log_lock:
        _request_log.append({"endpoint": endpoint, "details": details})
        if len(_request_log) > _MAX_LOG_ENTRIES:
            del _request_log[: -_MAX_LOG_ENTRIES]


def get_request_logs() -> list[dict[str, Any]]:
    """Thread-safe retrieval of request logs."""
    with _log_lock:
        return list(reversed(_request_log))


# ---------------------------------------------------------------------------
# Model validation helpers
# ---------------------------------------------------------------------------
def validate_sam3_model(weights_path: str) -> tuple[bool, str]:
    if not weights_path:
        return False, "Weights path is empty"
    path = Path(weights_path)
    if not path.exists():
        return False, f"SAM3 model not found at: {weights_path}"
    if not path.is_file():
        return False, f"Path is not a file: {weights_path}"
    return True, f"SAM3 model found ({path.stat().st_size / (1024 * 1024):.2f} MB)"


def validate_llm_model(model_path: str) -> tuple[bool, str]:
    if not model_path:
        return False, "Model path is empty"
    path = Path(model_path)
    if not path.exists():
        return False, f"Model not found at: {model_path}"
    if not path.is_file():
        return False, f"Path is not a file: {model_path}"
    return True, f"LLM model found"


def validate_opencv() -> tuple[bool, str]:
    try:
        import cv2  # noqa: F401
        return True, "OpenCV is available"
    except ImportError:
        return False, "OpenCV is not installed. Run: pip install opencv-python"


# ---------------------------------------------------------------------------
# Host API Request Handler (Background Server)
# ---------------------------------------------------------------------------
class HostAPIHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for Host API endpoints."""

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json_response(self, data: dict[str, Any], status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _read_body(self) -> Optional[bytes]:
        """Read request body with size limit."""
        raw_length = self.headers.get("Content-Length", "0")
        try:
            content_length = int(raw_length)
        except (ValueError, TypeError):
            self._send_json_response({"error": "Invalid Content-Length"}, status=400)
            return None

        if content_length > _MAX_REQUEST_BODY:
            self._send_json_response(
                {"error": {"code": "request_too_large", "message": f"Request body exceeds {_MAX_REQUEST_BODY} bytes"}},
                status=413,
            )
            return None

        return self.rfile.read(content_length)

    def _decode_image(self, frame_b64: str) -> Optional[np.ndarray]:
        """Decode a base64-encoded image with validation."""
        try:
            raw = base64.b64decode(frame_b64)
        except Exception:
            self._send_json_response({"error": {"code": "invalid_image", "message": "Invalid base64 encoding"}}, status=400)
            return None

        try:
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                self._send_json_response({"error": {"code": "invalid_image", "message": "Failed to decode image"}}, status=400)
                return None
            # Validate dimensions
            h, w = frame.shape[:2]
            if h > 10000 or w > 10000 or h * w > 50_000_000:
                self._send_json_response(
                    {"error": {"code": "image_too_large", "message": f"Image dimensions {w}x{h} exceed limits"}},
                    status=400,
                )
                return None
            return frame
        except Exception as exc:
            self._send_json_response({"error": {"code": "decode_failed", "message": str(exc)}}, status=400)
            return None

    def do_GET(self) -> None:
        """Handle GET requests (status endpoint)."""
        if self.path == "/api/host/status/":
            self._handle_status()
        else:
            self._send_json_response(
                {"error": {"code": "endpoint_not_found", "message": f"Unknown endpoint: {self.path}"}},
                status=404,
            )

    def do_POST(self) -> None:
        """Handle POST requests."""
        body = self._read_body()
        if body is None:
            return

        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json_response({"error": {"code": "invalid_json", "message": "Invalid JSON"}}, status=400)
            return

        endpoint = self.path
        if endpoint == "/api/host/evaluate-llm/":
            self._handle_evaluate_llm(data)
        elif endpoint == "/api/host/evaluate-sam3/":
            self._handle_evaluate_sam3(data)
        else:
            self._send_json_response(
                {"error": {"code": "endpoint_not_found", "message": f"Unknown endpoint: {self.path}"}},
                status=404,
            )

    def _handle_status(self) -> None:
        """Return Host listener + service status."""
        with _config_lock:
            config = dict(_host_api_config)

        sm = get_service_manager()
        llm_status = sm.status("host:llm")
        sam_status = sm.status("host:sam3")

        self._send_json_response({
            "status": "running" if _host_api_running else "stopped",
            "listen_ip": config.get("listen_ip", ""),
            "listen_port": config.get("listen_port", 8080),
            "llm": llm_status,
            "sam3": sam_status,
        })

    def _handle_evaluate_llm(self, data: dict[str, Any]) -> None:
        """Handle LLM evaluation using Host-owned, persistent config."""
        prompt = data.get("prompt", "")
        if not prompt:
            self._send_json_response(
                {"error": {"code": "missing_prompt", "message": "prompt is required"}},
                status=400,
            )
            return

        sm = get_service_manager()
        llm_status = sm.status("host:llm")

        if llm_status["state"] == "external":
            # Forward to external server
            import requests  # noqa: PLC0415
            store = get_settings_store()
            settings = store.load()
            llm_cfg = settings.host.llm
            try:
                resp = requests.post(
                    f"{llm_cfg.base_url}/completion",
                    json={"prompt": prompt},
                    timeout=llm_cfg.request_timeout_seconds,
                )
                resp.raise_for_status()
                result = resp.json()
                content = result.get("content", result.get("generation", ""))
                log_request("/api/host/evaluate-llm/", {"prompt_length": len(prompt)})
                self._send_json_response({
                    "content": content,
                    "model_id": llm_cfg.model_id,
                    "usage": None,
                    "metadata": {},
                })
            except Exception as exc:
                logger.exception("External LLM request failed")
                self._send_json_response(
                    {"error": {"code": "remote_request_failed", "message": str(exc)}},
                    status=502,
                )
        elif llm_status["state"] == "running":
            # Use managed service
            from ..utils.model_host.llama_server_helper import LlamaServerHelper  # noqa: PLC0415
            store = get_settings_store()
            settings = store.load()
            llm_cfg = settings.host.llm
            helper = LlamaServerHelper(
                model_path=llm_cfg.model_path,
                host=llm_cfg.host,
                port=llm_cfg.port,
            )
            helper._server_ready = True  # Trust the service manager
            try:
                result = helper.evaluate(
                    prompt,
                    max_tokens=data.get("max_tokens", 512),
                    temperature=data.get("temperature", 0.7),
                )
                log_request("/api/host/evaluate-llm/", {"prompt_length": len(prompt)})
                self._send_json_response({
                    "content": result,
                    "model_id": llm_cfg.model_id,
                    "usage": None,
                    "metadata": {},
                })
            except Exception as exc:
                logger.exception("LLM evaluation error")
                self._send_json_response(
                    {"error": {"code": "service_error", "message": str(exc)}},
                    status=500,
                )
        else:
            self._send_json_response(
                {"error": {"code": "service_not_running", "message": "Host LLM service is not running"}},
                status=503,
            )

    def _handle_evaluate_sam3(self, data: dict[str, Any]) -> None:
        """Handle SAM3 segmentation using Host-owned persistent config."""
        frame_b64 = data.get("frame_b64")
        if not frame_b64:
            self._send_json_response(
                {"error": {"code": "missing_frame", "message": "frame_b64 is required"}},
                status=400,
            )
            return

        frame = self._decode_image(frame_b64)
        if frame is None:
            return

        input_points = data.get("input_points")
        input_boxes = data.get("input_boxes")

        # Use Host-owned config
        sm = get_service_manager()
        sam_status = sm.status("host:sam3")

        if sam_status["state"] in ("stopped", "failed"):
            # Try to use the singleton SAM3 helper with Host config
            from ..utils.model_host.sam3_server_helper import Sam3ServerHelper  # noqa: PLC0415

            store = get_settings_store()
            settings = store.load()
            weights_path = settings.host.sam3.weights_path

            if not weights_path:
                self._send_json_response(
                    {"error": {"code": "model_not_configured", "message": "SAM3 weights path not configured on Host"}},
                    status=400,
                )
                return

            helper = Sam3ServerHelper(checkpoint_path=weights_path)
            if not helper.initialize():
                self._send_json_response(
                    {"error": {"code": "service_start_failed", "message": "Failed to initialize SAM3"}},
                    status=503,
                )
                return

            try:
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
                self._send_json_response(
                    {"error": {"code": "service_error", "message": str(exc)}},
                    status=500,
                )
        else:
            self._send_json_response(
                {"error": {"code": "service_not_running", "message": "SAM3 service not available"}},
                status=503,
            )


# ---------------------------------------------------------------------------
# Django Views
# ---------------------------------------------------------------------------

def landing_page(request: HttpRequest) -> HttpResponse:
    """Landing page: Route to Host or Client portal."""
    return render(request, "core_orchestrator/index.html")


@csrf_exempt
def host_portal(request: HttpRequest) -> HttpResponse:
    """
    Host Portal: Configure listener, LLM, and SAM3.

    GET:  Render configuration from persistent settings.
    POST: Start/stop listener OR manage services.
    """
    store = get_settings_store()
    settings = store.load()

    if request.method == "GET":
        with _config_lock:
            config_snapshot = dict(_host_api_config)

        return render(request, "core_orchestrator/host_portal.html", {
            "listen_ip": config_snapshot.get("listen_ip", settings.host.listen_ip),
            "listen_port": config_snapshot.get("listen_port", settings.host.listen_port),
            "sam3_weights_path": config_snapshot.get("sam3_weights_path", settings.host.sam3.weights_path),
            "running": _host_api_running,
            "settings_json": json.dumps(_dataclass_to_dict(settings), indent=2),
        })

    # POST
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return _error("invalid_json", "Invalid JSON")

    action = body.get("action", "")

    # --- Listener actions ---
    if action == "stop_listener":
        return _stop_listener()

    if action == "start_listener":
        return _start_listener(body, settings)

    if action == "restart_listener":
        stop_result = _stop_listener()
        if stop_result.status_code >= 400 and json.loads(stop_result.content).get("error", {}).get("code") != "already_stopped":
            return stop_result
        return _start_listener(body, settings)

    # --- Service management actions ---
    if action in ("start_service", "stop_service", "restart_service"):
        return _handle_service_action(action, body, settings)

    # --- Save settings ---
    if action == "save_settings":
        try:
            settings = store.update(body.get("settings", {}))
            return JsonResponse({"status": "ok", "settings": _dataclass_to_dict(settings)})
        except Exception as exc:
            logger.exception("Save failed")
            return _error("settings_save_failed", str(exc), status=500)

    return _error("invalid_action", f"Unknown action: {action}")


def _stop_listener() -> JsonResponse:
    """Stop the Host API listener with lifecycle protection."""
    global _host_api_running, _host_api_thread, _host_api_server

    with _lifecycle_lock:
        if not _host_api_running:
            return _error("already_stopped", "Listener is not running", status=409)

        if _host_api_server is not None:
            try:
                _host_api_server.shutdown()
                _host_api_server.server_close()
            except Exception as exc:
                logger.warning("Error shutting down HTTP server: %s", exc)

        if _host_api_thread is not None:
            _host_api_thread.join(timeout=5)
            if _host_api_thread.is_alive():
                logger.error("Host API server thread did not stop within 5s")

        _host_api_running = False
        _host_api_thread = None
        _host_api_server = None

    return JsonResponse({"status": "ok", "message": "Listener stopped"})


def _start_listener(body: dict, settings: AppSettings) -> JsonResponse:
    """Start the Host API listener with lifecycle protection."""
    global _host_api_running, _host_api_thread, _host_api_server

    listen_ip = body.get("listen_ip", settings.host.listen_ip)
    listen_port = int(body.get("listen_port", settings.host.listen_port))

    with _lifecycle_lock:
        if _host_api_running:
            return _error("already_running", "Listener is already running", status=409)

        with _config_lock:
            _host_api_config["listen_ip"] = listen_ip
            _host_api_config["listen_port"] = listen_port
            _host_api_config["sam3_weights_path"] = settings.host.sam3.weights_path

        try:
            server = ThreadingHTTPServer((listen_ip, listen_port), HostAPIHandler)
            _host_api_server = server
            _host_api_thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
                name="host-api-server",
            )
            _host_api_thread.start()
            _host_api_running = True
            logger.info("Host API listener started on %s:%d", listen_ip, listen_port)
            return JsonResponse({
                "status": "ok",
                "message": "Listener started",
                "listen_ip": listen_ip,
                "listen_port": listen_port,
            })
        except Exception as exc:
            logger.exception("Failed to start listener")
            _host_api_server = None
            _host_api_thread = None
            _host_api_running = False
            return _error("listener_start_failed", str(exc), status=500)


def _handle_service_action(action: str, body: dict, settings: AppSettings) -> JsonResponse:
    """Handle start/stop/restart for host:llm or host:sam3."""
    service_id = body.get("service_id", "")
    if service_id not in ("host:llm", "host:sam3"):
        return _error("invalid_service_id", f"Unknown service: {service_id}")

    sm = get_service_manager()

    if action == "start_service":
        cfg = settings.host.llm if service_id == "host:llm" else settings.host.sam3
        result = sm.start(service_id, cfg)
        return JsonResponse(result)

    if action == "stop_service":
        result = sm.stop(service_id)
        return JsonResponse(result)

    if action == "restart_service":
        cfg = settings.host.llm if service_id == "host:llm" else settings.host.sam3
        result = sm.restart(service_id, cfg)
        return JsonResponse(result)

    return _error("invalid_action", f"Unknown action: {action}")


# ---------------------------------------------------------------------------
# Legacy/backward-compat views (imported from old views.py)
# ---------------------------------------------------------------------------

@csrf_exempt
def host_evaluate_llm(request: HttpRequest) -> JsonResponse:
    """
    Host API Endpoint: Run LLM evaluation using Host-owned persistent config.
    No longer requires model_path from client.
    """
    if request.method != "POST":
        return _error("method_not_allowed", "Method not allowed", status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return _error("invalid_json", "Invalid JSON")

    prompt = body.get("prompt")
    if not prompt:
        return _error("missing_prompt", "prompt is required")

    store = get_settings_store()
    settings = store.load()
    llm_cfg = settings.host.llm

    sm = get_service_manager()
    llm_status = sm.status("host:llm")

    if llm_status["state"] == "running":
        from ..utils.model_host.llama_server_helper import LlamaServerHelper
        helper = LlamaServerHelper(
            model_path=llm_cfg.model_path,
            host=llm_cfg.host,
            port=llm_cfg.port,
        )
        helper._server_ready = True
        try:
            result = helper.evaluate(prompt, max_tokens=body.get("max_tokens", 512))
            log_request("/api/host/evaluate-llm/", {"prompt_length": len(prompt)})
            return JsonResponse({"content": result, "model_id": llm_cfg.model_id, "usage": None, "metadata": {}})
        except Exception as exc:
            logger.exception("LLM evaluation error")
            return _error("service_error", str(exc), status=500)
    elif llm_status["state"] == "external":
        import requests
        try:
            resp = requests.post(
                f"{llm_cfg.base_url}/completion",
                json={"prompt": prompt},
                timeout=llm_cfg.request_timeout_seconds,
            )
            resp.raise_for_status()
            result = resp.json()
            content = result.get("content", result.get("generation", ""))
            return JsonResponse({"content": content, "model_id": llm_cfg.model_id, "usage": None, "metadata": {}})
        except Exception as exc:
            logger.exception("External LLM request failed")
            return _error("remote_request_failed", str(exc), status=502)
    else:
        return _error("service_not_running", "Host LLM service is not running", status=503)


@csrf_exempt
def host_evaluate_sam3(request: HttpRequest) -> JsonResponse:
    """
    Host API Endpoint: Run SAM3 segmentation using Host-owned config.
    Client no longer sends weights_path.
    """
    if request.method != "POST":
        return _error("method_not_allowed", "Method not allowed", status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return _error("invalid_json", "Invalid JSON")

    frame_b64 = body.get("frame_b64")
    if not frame_b64:
        return _error("missing_frame", "frame_b64 is required")

    try:
        raw = base64.b64decode(frame_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return _error("invalid_image", "Failed to decode frame")
        h, w = frame.shape[:2]
        if h > 10000 or w > 10000 or h * w > 50_000_000:
            return _error("image_too_large", f"Image dimensions {w}x{h} exceed limits")
    except Exception as exc:
        return _error("invalid_image", f"Invalid frame_b64: {exc}")

    input_points = body.get("input_points")
    input_boxes = body.get("input_boxes")

    store = get_settings_store()
    settings = store.load()
    weights_path = settings.host.sam3.weights_path

    if not weights_path:
        return _error("model_not_configured", "SAM3 weights path not configured on Host. Configure in Host Portal.")

    try:
        from ..utils.model_host.sam3_server_helper import Sam3ServerHelper
        helper = Sam3ServerHelper(checkpoint_path=weights_path)
        if not helper.initialize():
            return _error("service_start_failed", "Failed to initialize SAM3", status=503)

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
        return _error("service_error", str(exc), status=500)


def host_status(request: HttpRequest) -> JsonResponse:
    """Return Host status: listener + services."""
    with _config_lock:
        config_snapshot = dict(_host_api_config)

    sm = get_service_manager()
    llm_status = sm.status("host:llm")
    sam_status = sm.status("host:sam3")

    store = get_settings_store()
    saved = store.load()

    return JsonResponse({
        "status": "running" if _host_api_running else "stopped",
        "listen_ip": config_snapshot.get("listen_ip", saved.host.listen_ip),
        "listen_port": config_snapshot.get("listen_port", saved.host.listen_port),
        "llm": llm_status,
        "sam3": sam_status,
        "restart_required": config_snapshot.get("listen_ip", "") != saved.host.listen_ip
        or config_snapshot.get("listen_port", 0) != saved.host.listen_port,
    })


# ---------------------------------------------------------------------------
# Client views
# ---------------------------------------------------------------------------

def client_portal(request: HttpRequest) -> HttpResponse:
    """Client Portal: Tool selection page."""
    return render(request, "core_orchestrator/tool_selection.html")


def client_run_workspace(request: HttpRequest) -> HttpResponse:
    """Render a shared progress workspace for selected client tools."""
    tool_catalog = {
        "drone-heatmap": {
            "label": "Drone Heatmap",
            "description": "Generating heatmap views from selected drone captures.",
            "steps": ["Load source media", "Run heatmap inference", "Build selected views", "Publish results"],
            "settings_key": "heatmap_views",
        },
        "knowledge-graph": {
            "label": "Knowledge Graph",
            "description": "Extracting entities, relationships, and scene context.",
            "steps": ["Scan source media", "Detect objects", "Map relationships", "Export graph"],
            "settings_key": "graph_settings",
        },
        "3d-reconstruction": {
            "label": "3D Reconstruction",
            "description": "Building a spatial reconstruction from selected captures.",
            "steps": ["Index frames", "Match features", "Estimate scene geometry", "Render reconstruction"],
            "settings_key": "reconstruction_settings",
        },
    }

    requested_tools = [
        tool for tool in request.GET.get("tools", "").split(",") if tool in tool_catalog
    ]
    if not requested_tools:
        requested_tools = ["drone-heatmap"]

    selected_tools = []
    for tool_key in requested_tools:
        tool = dict(tool_catalog[tool_key])
        tool["key"] = tool_key
        tool["settings"] = request.GET.get(tool["settings_key"], "")
        tool["file_count"] = request.GET.get("file_count", "0")
        selected_tools.append(tool)

    return render(
        request,
        "core_orchestrator/client_run_workspace.html",
        {"selected_tools": selected_tools},
    )


def heatmap_dashboard(request: HttpRequest) -> HttpResponse:
    """
    Heatmap Dashboard: Main client interface.
    Uses persistent settings with independent LLM/SAM routing.
    """
    store = get_settings_store()
    settings = store.load()

    config = {
        "dataset_path": settings.client.dataset_path or request.GET.get("dataset_path", ""),
        "llm_mode": settings.client.llm_mode,
        "sam3_mode": settings.client.sam3_mode,
        "remote_host_ip": settings.client.remote_host.host,
        "remote_host_port": settings.client.remote_host.port,
        "settings_json": json.dumps(_dataclass_to_dict(settings), indent=2),
    }
    return render(request, "core_orchestrator/heatmap_dashboard.html", {"config": config})


def heatmap_stream(request: HttpRequest) -> StreamingHttpResponse:
    """
    MJPEG stream endpoint.
    Uses independent llm_mode and sam3_mode. Accepts legacy routing_mode.
    """
    dataset_path = request.GET.get("dataset_path", "")
    llm_mode = request.GET.get("llm_mode", "")
    sam3_mode = request.GET.get("sam3_mode", "")
    legacy_mode = request.GET.get("routing_mode", "")

    # Backward compat: if only routing_mode is set, use it for both
    if not llm_mode and not sam3_mode and legacy_mode:
        llm_mode = legacy_mode
        sam3_mode = legacy_mode

    # Default from settings if still empty
    if not llm_mode or not sam3_mode:
        store = get_settings_store()
        settings = store.load()
        if not llm_mode:
            llm_mode = settings.client.llm_mode
        if not sam3_mode:
            sam3_mode = settings.client.sam3_mode

    remote_host_ip = request.GET.get("remote_host_ip", "127.0.0.1")
    remote_host_port = request.GET.get("remote_host_port", "8080")
    sam3_weights_path = request.GET.get("sam3_weights_path", "")

    def create_error_frame(error_msg: str, width: int = 640, height: int = 480) -> bytes:
        try:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
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
            y_offset = 50
            for i, line in enumerate(lines[:5]):
                y_pos = y_offset + i * 30
                cv2.putText(frame, line, (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 50), 2)
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buffer.tobytes()
        except Exception:
            fallback = np.full((480, 640, 3), (0, 0, 200), dtype=np.uint8)
            _, buffer = cv2.imencode(".jpg", fallback, [cv2.IMWRITE_JPEG_QUALITY, 50])
            return buffer.tobytes()

    def generate():
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
            logger.info("No dataset path provided, showing placeholder")
            placeholder_frame = np.full((480, 640, 3), (50, 50, 60), dtype=np.uint8)
            cv2_local.putText(placeholder_frame, "No video source configured", (180, 200),
                             cv2_local.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2_local.putText(placeholder_frame, "Set dataset path in Heatmap Dashboard", (140, 240),
                             cv2_local.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            _, buffer = cv2_local.imencode(".jpg", placeholder_frame, [cv2_local.IMWRITE_JPEG_QUALITY, 85])
            while True:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )
                return

        cap.set(cv2_local.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2_local.CAP_PROP_FRAME_HEIGHT, 480)

        sam3_helper = None
        remote_client = None

        try:
            if sam3_mode == "remote":
                from ..utils.model_host.remote_client_helper import RemoteClientHelper
                remote_client = RemoteClientHelper(
                    base_url=f"http://{remote_host_ip}:{remote_host_port}"
                )
            else:
                from ..utils.model_host.sam3_server_helper import Sam3ServerHelper
                if not sam3_weights_path:
                    store = get_settings_store()
                    settings = store.load()
                    sam3_weights_path = settings.client.local_sam3.weights_path
                sam3_helper = Sam3ServerHelper(checkpoint_path=sam3_weights_path)
                sam3_helper.initialize()

            frame_count = 0
            skip_frames = 2

            while True:
                ret, frame = cap.read()
                if not ret:
                    if dataset_path:
                        cap.set(cv2_local.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        error_bytes = create_error_frame("Camera stream ended")
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + error_bytes + b"\r\n"
                        )
                        break

                if frame_count % skip_frames != 0:
                    _, buffer = cv2_local.imencode(".jpg", frame, [cv2_local.IMWRITE_JPEG_QUALITY, 85])
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                    )
                    frame_count += 1
                    continue

                frame_count += 1

                try:
                    if sam3_mode == "remote":
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
                        if sam3_helper:
                            result = sam3_helper.predict(frame)
                            masks = result.get("masks", [])
                            target_coords = result.get("target_coords", [])
                            scores = result.get("scores", [])
                        else:
                            masks = []
                            target_coords = []
                            scores = []

                    if masks and len(masks) > 0:
                        for i, mask in enumerate(masks):
                            if len(mask) > 0 and isinstance(mask[0], list):
                                try:
                                    import numpy as np
                                    mask_arr = np.array(mask, dtype=np.uint8)
                                    overlay = frame.copy()
                                    contours, _ = cv2_local.findContours(
                                        mask_arr, cv2_local.RETR_EXTERNAL, cv2_local.CHAIN_APPROX_SIMPLE
                                    )
                                    cv2_local.drawContours(overlay, contours, -1, (0, 255, 0), 2)
                                    for contour in contours:
                                        if len(contour) > 0:
                                            cv2_local.drawContours(overlay, [contour], -1, (100, 200, 100), -1)
                                    frame = cv2_local.addWeighted(overlay, 1, frame, 1, 0)
                                    if i < len(target_coords):
                                        cx, cy = target_coords[i]
                                        cv2_local.circle(frame, (int(cx), int(cy)), 8, (255, 0, 0), -1)
                                        if i < len(scores):
                                            score_text = f"{scores[i]:.2f}"
                                            cv2_local.putText(frame, score_text, (int(cx) - 30, int(cy) - 15),
                                                             cv2_local.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                                except Exception as mask_exc:
                                    logger.warning("Failed to process mask: %s", mask_exc)
                                    continue

                    mode_text = f"SAM: {sam3_mode.upper()}"
                    cv2_local.putText(frame, mode_text, (10, 30),
                                     cv2_local.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    frame_text = f"Frame: {frame_count}"
                    cv2_local.putText(frame, frame_text, (10, 60),
                                     cv2_local.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                except Exception as infer_exc:
                    logger.warning("Inference error on frame %d: %s", frame_count, infer_exc)

                _, buffer = cv2_local.imencode(".jpg", frame, [cv2_local.IMWRITE_JPEG_QUALITY, 85])
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
                )

        except Exception as exc:
            logger.exception("Stream error: %s", exc)
            error_bytes = create_error_frame(f"Stream Error: {exc}")
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + error_bytes + b"\r\n"
            )
        finally:
            if cap:
                cap.release()

    return StreamingHttpResponse(
        generate(),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


# Ensure Path is available for validation functions
from pathlib import Path  # noqa: E402, F811