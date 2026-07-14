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
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
import threading
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import PurePosixPath
from typing import Any, Generator, Optional

import cv2
import numpy as np

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)


def _shell_command_text(command: list[str]) -> str:
    return shlex.join(command)

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

# MASt3R-SLAM integration. Prefer the known-good local checkout used by
# Not-Really-ARCADIA, then fall back to the vendored copy for cloned installs.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VENDORED_MAST3R_SLAM_ROOT = os.path.join(PROJECT_ROOT, "vendor", "mast3r-slam")
LOCAL_MAST3R_SLAM_ROOT = os.path.join(PROJECT_ROOT, "MASt3R-SLAM")
CANONICAL_MAST3R_SLAM_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "MASt3R-SLAM"))
DEFAULT_MAST3R_SLAM_ROOT = (
    LOCAL_MAST3R_SLAM_ROOT
    if os.path.isfile(os.path.join(LOCAL_MAST3R_SLAM_ROOT, "main.py"))
    else CANONICAL_MAST3R_SLAM_ROOT
    if os.path.isfile(os.path.join(CANONICAL_MAST3R_SLAM_ROOT, "main.py"))
    else VENDORED_MAST3R_SLAM_ROOT
)
MAST3R_SLAM_ROOT = os.environ.get("MAST3R_SLAM_ROOT", DEFAULT_MAST3R_SLAM_ROOT)
_mast3r_local_python = os.path.join(MAST3R_SLAM_ROOT, ".venv", "bin", "python")
MAST3R_SLAM_PYTHON = os.environ.get(
    "MAST3R_SLAM_PYTHON",
    _mast3r_local_python if os.path.isfile(_mast3r_local_python) else sys.executable,
)
MAST3R_SLAM_MAIN = os.path.join(MAST3R_SLAM_ROOT, "main.py")
LINGBOT_MAP_ROOT = os.environ.get("LINGBOT_MAP_ROOT", os.path.join(PROJECT_ROOT, "lingbot-map"))
DEFAULT_LINGBOT_MAP_PYTHON = "/home/jtate60/miniforge3/envs/lingbot-map/bin/python3.10"
LINGBOT_MAP_PYTHON = os.environ.get(
    "LINGBOT_MAP_PYTHON",
    DEFAULT_LINGBOT_MAP_PYTHON if os.path.isfile(DEFAULT_LINGBOT_MAP_PYTHON) else sys.executable,
)
LINGBOT_MAP_MAIN = os.path.join(LINGBOT_MAP_ROOT, "lingbot_live_demo.py")
LINGBOT_MAP_MODEL = os.environ.get("LINGBOT_MAP_MODEL", os.path.join(LINGBOT_MAP_ROOT, "lingbot-map.pt"))
DRONE_3D_RECONSTRUCTION_ROOT = os.environ.get(
    "DRONE_3D_RECONSTRUCTION_ROOT",
    os.path.join(PROJECT_ROOT, "drone_3d_reconstruction"),
)
DRONE_3D_RECONSTRUCTION_SRC = os.path.join(DRONE_3D_RECONSTRUCTION_ROOT, "src")
DRONE_3D_RECONSTRUCTION_PYTHON = os.environ.get(
    "DRONE_3D_RECONSTRUCTION_PYTHON",
    MAST3R_SLAM_PYTHON if os.path.isfile(MAST3R_SLAM_PYTHON) else sys.executable,
)
_mast3r_process: Optional[subprocess.Popen[str]] = None
_mast3r_run: Optional[dict[str, Any]] = None
_mast3r_lock = threading.Lock()

UPLOAD_ROOT = os.path.join(settings.BASE_DIR, "uploads", "reconstruction")
MAST3R_RUNS_ROOT = os.path.join(settings.BASE_DIR, "runtime", "mast3r_runs")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


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


def _mast3r_status_payload() -> dict[str, Any]:
    """Return the current reconstruction launcher state."""
    running = _mast3r_process is not None and _mast3r_process.poll() is None
    return_code = _mast3r_process.poll() if _mast3r_process is not None else None
    engine = _mast3r_run.get("engine", "mast3r-slam") if _mast3r_run else "mast3r-slam"
    discovered_pid = _discover_reconstruction_pid(engine)
    if not running and discovered_pid is None and _mast3r_run is None:
        for fallback_engine in ("lingbot-map", "colmap-drone"):
            fallback_pid = _discover_reconstruction_pid(fallback_engine)
            if fallback_pid is not None:
                engine = fallback_engine
                discovered_pid = fallback_pid
                break
    if not running and discovered_pid is not None:
        running = True

    status = "running" if running else "stopped"
    if _mast3r_process is not None and return_code is not None and discovered_pid is None:
        status = "completed" if return_code == 0 else f"failed ({return_code})"

    payload = {
        "running": running,
        "pid": _mast3r_process.pid if running and _mast3r_process else discovered_pid,
        "root": _mast3r_run.get("root") if _mast3r_run else _engine_root(engine),
        "status": status,
        "engine": engine,
    }
    if _mast3r_run:
        viewer_url = _mast3r_run.get("viewer_url", "")
        artifact_url = ""
        if _mast3r_run.get("engine") == "colmap-drone":
            run_id = _mast3r_run.get("run_id", "")
            dense_ply_path = _colmap_dense_ply_path(run_id)
            if dense_ply_path and os.path.isfile(dense_ply_path):
                viewer_url = f"/api/reconstruction/viewer/{run_id}/"
                artifact_url = f"/api/reconstruction/artifact/{run_id}/dense.ply"
        payload.update({
            "engine": _mast3r_run.get("engine", "mast3r-slam"),
            "run_id": _mast3r_run.get("run_id"),
            "dataset": _mast3r_run.get("dataset"),
            "command": " ".join(_mast3r_run.get("command", [])),
            "log_path": _mast3r_run.get("log_path"),
            "log_tail": _read_log_tail(_mast3r_run.get("log_path", "")),
            "viewer_url": viewer_url,
            "artifact_url": artifact_url,
        })
    else:
        log_path = _latest_mast3r_log_path()
        if log_path:
            payload.update({
                "log_path": log_path,
                "log_tail": _read_log_tail(log_path),
            })
    return payload


def _run_dir_for_id(run_id: str) -> str:
    safe_run_id = _slugify(run_id)
    run_dir = os.path.abspath(os.path.join(MAST3R_RUNS_ROOT, safe_run_id))
    runs_root = os.path.abspath(MAST3R_RUNS_ROOT)
    if os.path.commonpath([runs_root, run_dir]) != runs_root:
        raise Http404("Invalid reconstruction run")
    return run_dir


def _colmap_dense_ply_path(run_id: str) -> str:
    if not run_id:
        return ""
    return os.path.join(_run_dir_for_id(run_id), "colmap_output", "dense.ply")


def _engine_label(engine: str) -> str:
    if engine == "colmap-drone":
        return "COLMAP 3D Reconstruction"
    return "LingBot-Map" if engine == "lingbot-map" else "MASt3R-SLAM"


def _engine_root(engine: str) -> str:
    if engine == "colmap-drone":
        return DRONE_3D_RECONSTRUCTION_ROOT
    return LINGBOT_MAP_ROOT if engine == "lingbot-map" else MAST3R_SLAM_ROOT


def _discover_reconstruction_pid(engine: str) -> Optional[int]:
    if engine == "lingbot-map":
        return _discover_lingbot_pid()
    if engine == "colmap-drone":
        return _discover_colmap_drone_pid()
    return _discover_mast3r_pid()


def _discover_mast3r_pid() -> Optional[int]:
    """Find a running vendored MASt3R-SLAM main process if Django lost its handle."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"{MAST3R_SLAM_ROOT}.+main.py"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    for line in result.stdout.splitlines():
        try:
            return int(line.strip())
        except ValueError:
            continue
    return None


def _discover_lingbot_pid() -> Optional[int]:
    """Find a running LingBot-Map live demo if Django lost its handle."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"{LINGBOT_MAP_ROOT}.+lingbot_live_demo.py"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid != current_pid:
            return pid
    return None


def _discover_colmap_drone_pid() -> Optional[int]:
    """Find a running COLMAP drone reconstruction launched from this checkout."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"{DRONE_3D_RECONSTRUCTION_ROOT}.+colmap_reconstruction.orchestrate"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    current_pid = os.getpid()
    for line in result.stdout.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid != current_pid:
            return pid
    return None


def _latest_mast3r_log_path() -> str:
    log_paths = []
    for root, _dirs, files in os.walk(MAST3R_RUNS_ROOT):
        if "run.log" in files:
            path = os.path.join(root, "run.log")
            try:
                log_paths.append((os.path.getmtime(path), path))
            except OSError:
                continue
    return max(log_paths, default=(0.0, ""))[1]


def _read_log_tail(log_path: str, max_lines: int = 80) -> str:
    if not log_path or not os.path.exists(log_path):
        return ""
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as log_file:
            return "\n".join(log_file.read().splitlines()[-max_lines:])
    except OSError:
        return ""


def _slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value.strip())
    return cleaned.strip("-_") or "web-launch"


def _is_video_dataset(dataset: str) -> bool:
    return os.path.isfile(dataset) and os.path.splitext(dataset)[1].lower() in VIDEO_EXTENSIONS


def _stop_mast3r_process(process: subprocess.Popen[str], engine: str = "mast3r-slam") -> None:
    if process.poll() is not None:
        _kill_reconstruction_descendants(engine)
        return
    try:
        if sys.platform != "win32":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        process.wait(timeout=5)
    except Exception:
        try:
            if sys.platform != "win32":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                if sys.platform != "win32":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=3)
            except Exception:
                logger.exception("Failed to stop %s process group", _engine_label(engine))
    finally:
        _kill_reconstruction_descendants(engine)


def _stop_mast3r_pid(pid: int, engine: str = "mast3r-slam") -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            if sys.platform != "win32":
                try:
                    os.killpg(pid, sig)
                except ProcessLookupError:
                    os.kill(pid, sig)
            else:
                os.kill(pid, sig)
            time.sleep(1)
            break
        except Exception:
            if sig == signal.SIGTERM:
                logger.exception("Failed to stop discovered %s process", _engine_label(engine))
    _kill_reconstruction_descendants(engine)


def _kill_reconstruction_descendants(engine: str) -> None:
    if engine == "lingbot-map":
        _kill_lingbot_descendants()
    elif engine == "colmap-drone":
        _kill_colmap_drone_descendants()
    else:
        _kill_mast3r_descendants()


def _kill_mast3r_descendants() -> None:
    """Clean up MASt3R-SLAM worker processes left after the main process exits."""
    if sys.platform == "win32":
        return
    patterns = [
        f"{MAST3R_SLAM_ROOT}/.venv/bin/python -c from multiprocessing",
        f"{MAST3R_SLAM_ROOT}.+main.py",
    ]
    for sig in ("TERM", "KILL"):
        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", f"-{sig}", "-f", pattern],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                continue
        if sig == "TERM":
            time.sleep(1)


def _kill_lingbot_descendants() -> None:
    """Clean up LingBot-Map live viewer/inference processes from this checkout."""
    if sys.platform == "win32":
        return
    patterns = [
        f"{LINGBOT_MAP_ROOT}.+lingbot_live_demo.py",
        f"{LINGBOT_MAP_PYTHON}.+lingbot_live_demo.py",
    ]
    for sig in ("TERM", "KILL"):
        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", f"-{sig}", "-f", pattern],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                continue
        if sig == "TERM":
            time.sleep(1)


def _kill_colmap_drone_descendants() -> None:
    """Clean up COLMAP drone reconstruction processes from this checkout."""
    if sys.platform == "win32":
        return
    patterns = [
        f"{DRONE_3D_RECONSTRUCTION_ROOT}.+colmap_reconstruction.orchestrate",
    ]
    for sig in ("TERM", "KILL"):
        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", f"-{sig}", "-f", pattern],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except OSError:
                continue
        if sig == "TERM":
            time.sleep(1)


@csrf_exempt
def reconstruction_start(request: HttpRequest) -> JsonResponse:
    """Start the selected live reconstruction workflow."""
    global _mast3r_process, _mast3r_run

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    with _mast3r_lock:
        if _mast3r_process is not None and _mast3r_process.poll() is None:
            return JsonResponse({
                "message": "A reconstruction process is already running",
                **_mast3r_status_payload(),
            })

        try:
            body = json.loads(request.body or b"{}")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        dataset = body.get("dataset", "").strip()
        config = body.get("config", "config/base.yaml")
        engine = body.get("engine", "mast3r-slam")
        save_as = _slugify(body.get("save_as", "web-launch"))
        if not dataset:
            return JsonResponse({"error": "Dataset path is required"}, status=400)
        if not os.path.exists(dataset):
            return JsonResponse({"error": f"Dataset path not found: {dataset}"}, status=400)
        if engine not in {"mast3r-slam", "lingbot-map", "colmap-drone"}:
            return JsonResponse({"error": f"Unknown reconstruction engine: {engine}"}, status=400)

        stale_pid = _discover_reconstruction_pid(engine)
        if stale_pid is not None:
            _stop_mast3r_pid(stale_pid, engine)

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}-{save_as}"
        run_dir = os.path.join(MAST3R_RUNS_ROOT, run_id)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        viewer_url = ""

        process_env = None

        if engine == "lingbot-map":
            if not os.path.isdir(LINGBOT_MAP_ROOT):
                return JsonResponse({"error": f"LingBot-Map root not found: {LINGBOT_MAP_ROOT}"}, status=500)
            if not os.path.isfile(LINGBOT_MAP_PYTHON):
                return JsonResponse({"error": f"LingBot-Map Python not found: {LINGBOT_MAP_PYTHON}"}, status=500)
            if not os.path.isfile(LINGBOT_MAP_MAIN):
                return JsonResponse({"error": f"LingBot-Map live demo not found: {LINGBOT_MAP_MAIN}"}, status=500)
            if not os.path.isfile(LINGBOT_MAP_MODEL):
                return JsonResponse({"error": f"LingBot-Map model not found: {LINGBOT_MAP_MODEL}"}, status=500)

            port = int(body.get("port", 8080))
            command = [
                LINGBOT_MAP_PYTHON,
                LINGBOT_MAP_MAIN,
                "--model_path",
                LINGBOT_MAP_MODEL,
                #"--stride",
                #"10",
                "--mode",
                "windowed",
                "--window_size",
                "32",
                "--overlap_size",
                "8",
                "--keyframe_interval",
                "1",
                "--use_sdpa",
                "--downsample_factor",
                "1",
                "--conf_threshold",
                "1.5",
                "--port",
                str(port),
            ]
            command.extend(["--video_path" if _is_video_dataset(dataset) else "--image_folder", dataset])
            cwd = LINGBOT_MAP_ROOT
            root = LINGBOT_MAP_ROOT
            config_path = ""
            viewer_url = f"http://localhost:{port}/?run_id={run_id}"
        elif engine == "colmap-drone":
            if not os.path.isdir(DRONE_3D_RECONSTRUCTION_ROOT):
                return JsonResponse({"error": f"COLMAP drone reconstruction root not found: {DRONE_3D_RECONSTRUCTION_ROOT}"}, status=500)
            if not os.path.isdir(DRONE_3D_RECONSTRUCTION_SRC):
                return JsonResponse({"error": f"COLMAP drone reconstruction src not found: {DRONE_3D_RECONSTRUCTION_SRC}"}, status=500)
            if not shutil.which("colmap") and not os.path.exists(os.path.join(DRONE_3D_RECONSTRUCTION_ROOT, "tools", "colmap", "COLMAP.bat")):
                return JsonResponse({"error": "COLMAP executable not found. Install COLMAP or set COLMAP_EXE."}, status=500)

            output_dir = os.path.join(run_dir, "colmap_output")
            command = [
                DRONE_3D_RECONSTRUCTION_PYTHON,
                "-m",
                "colmap_reconstruction.orchestrate",
                dataset,
                output_dir,
            ]
            cwd = DRONE_3D_RECONSTRUCTION_ROOT
            root = DRONE_3D_RECONSTRUCTION_ROOT
            config_path = ""
            process_env = os.environ.copy()
            existing_pythonpath = process_env.get("PYTHONPATH", "")
            process_env["PYTHONPATH"] = (
                DRONE_3D_RECONSTRUCTION_SRC
                if not existing_pythonpath
                else f"{DRONE_3D_RECONSTRUCTION_SRC}{os.pathsep}{existing_pythonpath}"
            )
        else:
            if not os.path.isdir(MAST3R_SLAM_ROOT):
                return JsonResponse({"error": f"MASt3R-SLAM root not found: {MAST3R_SLAM_ROOT}"}, status=500)
            if not os.path.isfile(MAST3R_SLAM_PYTHON):
                return JsonResponse({"error": f"MASt3R-SLAM Python not found: {MAST3R_SLAM_PYTHON}"}, status=500)
            if not os.path.isfile(MAST3R_SLAM_MAIN):
                return JsonResponse({"error": f"MASt3R-SLAM main.py not found: {MAST3R_SLAM_MAIN}"}, status=500)
            config_path = config if os.path.isabs(config) else os.path.join(MAST3R_SLAM_ROOT, config)
            if not os.path.isfile(config_path):
                return JsonResponse({"error": f"MASt3R-SLAM config not found: {config_path}"}, status=400)

            command = [
                MAST3R_SLAM_PYTHON,
                MAST3R_SLAM_MAIN,
                "--dataset",
                dataset,
                "--config",
                config_path,
                "--save-as",
                save_as,
            ]
            cwd = MAST3R_SLAM_ROOT
            root = MAST3R_SLAM_ROOT

        log_file = open(log_path, "w", encoding="utf-8")
        log_file.write("Command:\n")
        log_file.write(_shell_command_text(command) + "\n\n")
        log_file.flush()
        try:
            _mast3r_process = subprocess.Popen(
                command,
                cwd=cwd,
                env=process_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=(sys.platform != "win32"),
            )
            _mast3r_run = {
                "engine": engine,
                "run_id": run_id,
                "dataset": dataset,
                "config": config_path,
                "save_as": save_as,
                "command": command,
                "log_path": log_path,
                "run_dir": run_dir,
                "started_at": timestamp,
                "root": root,
                "viewer_url": viewer_url,
            }
        except Exception as exc:
            log_file.close()
            logger.exception("Failed to start reconstruction process")
            return JsonResponse({"error": str(exc)}, status=500)

    return JsonResponse({
        "message": f"{_engine_label(engine)} started",
        "command": _shell_command_text(command),
        "log_path": log_path,
        **_mast3r_status_payload(),
    })


@csrf_exempt
def reconstruction_stop(request: HttpRequest) -> JsonResponse:
    """Stop a reconstruction process launched by Almost-ARCADIA."""
    global _mast3r_process

    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    with _mast3r_lock:
        engine = _mast3r_run.get("engine", "mast3r-slam") if _mast3r_run else "mast3r-slam"
        label = _engine_label(engine)
        if _mast3r_process is None or _mast3r_process.poll() is not None:
            discovered_pid = _discover_reconstruction_pid(engine)
            if discovered_pid is None and _mast3r_run is None:
                for fallback_engine in ("lingbot-map", "colmap-drone"):
                    discovered_pid = _discover_reconstruction_pid(fallback_engine)
                    if discovered_pid is not None:
                        engine = fallback_engine
                        label = _engine_label(engine)
                        break
            if discovered_pid is not None:
                _stop_mast3r_pid(discovered_pid, engine)
                return JsonResponse({"message": f"{label} stopped", **_mast3r_status_payload()})
            _mast3r_process = None
            return JsonResponse({"message": f"{label} is not running", **_mast3r_status_payload()})

        _stop_mast3r_process(_mast3r_process, engine)
        _mast3r_process = None

    return JsonResponse({"message": f"{label} stopped", **_mast3r_status_payload()})


def reconstruction_status(request: HttpRequest) -> JsonResponse:
    """Return the current MASt3R-SLAM process status."""
    if request.method != "GET":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    return JsonResponse(_mast3r_status_payload())


def reconstruction_artifact(request: HttpRequest, run_id: str, filename: str) -> FileResponse:
    """Serve generated reconstruction artifacts for a completed run."""
    if filename != "dense.ply":
        raise Http404("Unknown reconstruction artifact")
    artifact_path = _colmap_dense_ply_path(run_id)
    if not artifact_path or not os.path.isfile(artifact_path):
        raise Http404("Reconstruction artifact not found")
    return FileResponse(open(artifact_path, "rb"), filename="dense.ply")


def reconstruction_viewer(request: HttpRequest, run_id: str) -> HttpResponse:
    """Render a browser PLY viewer for a completed COLMAP reconstruction."""
    artifact_path = _colmap_dense_ply_path(run_id)
    if not artifact_path or not os.path.isfile(artifact_path):
        raise Http404("Reconstruction artifact not found")
    return render(
        request,
        "core_orchestrator/reconstruction_ply_viewer.html",
        {
            "run_id": _slugify(run_id),
            "artifact_url": f"/api/reconstruction/artifact/{_slugify(run_id)}/dense.ply",
        },
    )


def _safe_upload_relative_path(raw_path: str) -> str:
    """Normalize a browser-provided relative path for storage under a run dir."""
    clean_path = PurePosixPath(raw_path.replace("\\", "/"))
    safe_parts = [
        part for part in clean_path.parts
        if part not in {"", ".", ".."} and "/" not in part and "\\" not in part
    ]
    return os.path.join(*safe_parts) if safe_parts else "upload"


def _natural_sort_key(value: str) -> list[Any]:
    """Sort frame names in human order, so frame_2 appears before frame_10."""
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value.replace("\\", "/"))
    ]


def _ordered_frame_name(index: int, raw_name: str) -> str:
    safe_name = os.path.basename(_safe_upload_relative_path(raw_name))
    stem, extension = os.path.splitext(safe_name)
    cleaned_stem = _slugify(stem)
    return f"{index:06d}__{cleaned_stem}{extension.lower()}"


@csrf_exempt
def reconstruction_upload_dataset(request: HttpRequest) -> JsonResponse:
    """Persist an uploaded image set, image folder, or one video for reconstruction."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    files = request.FILES.getlist("files")
    relative_paths = request.POST.getlist("relative_paths")
    if not files:
        return JsonResponse({"error": "No dataset files were uploaded"}, status=400)

    image_files = []
    video_files = []
    unsupported_files = []
    for index, uploaded_file in enumerate(files):
        raw_name = relative_paths[index] if index < len(relative_paths) else uploaded_file.name
        extension = os.path.splitext(raw_name)[1].lower()
        if extension in IMAGE_EXTENSIONS:
            image_files.append((index, uploaded_file, raw_name))
        elif extension in VIDEO_EXTENSIONS:
            video_files.append((index, uploaded_file, raw_name))
        else:
            unsupported_files.append(raw_name)

    if unsupported_files:
        return JsonResponse({"error": "Upload images, a folder of images, or one video."}, status=400)
    if image_files and video_files:
        return JsonResponse({"error": "Choose images or one video, not both."}, status=400)
    if len(video_files) > 1:
        return JsonResponse({"error": "Only one video can be uploaded per dataset."}, status=400)

    run_id = uuid.uuid4().hex
    run_dir = os.path.abspath(os.path.join(UPLOAD_ROOT, run_id))
    os.makedirs(run_dir, exist_ok=True)
    storage = FileSystemStorage(location=run_dir)

    saved_paths = []
    if video_files:
        _index, uploaded_file, raw_name = video_files[0]
        relative_path = _safe_upload_relative_path(raw_name)
        saved_name = storage.save(relative_path, uploaded_file)
        saved_paths.append(os.path.abspath(os.path.join(run_dir, saved_name)))
        dataset_path = saved_paths[0]
    else:
        ordered_dir = os.path.join(run_dir, "ordered_frames")
        os.makedirs(ordered_dir, exist_ok=True)
        ordered_storage = FileSystemStorage(location=ordered_dir)
        ordered_image_files = sorted(image_files, key=lambda item: (_natural_sort_key(item[2]), item[0]))
        for frame_index, (_original_index, uploaded_file, raw_name) in enumerate(ordered_image_files, start=1):
            saved_name = ordered_storage.save(_ordered_frame_name(frame_index, raw_name), uploaded_file)
            saved_paths.append(os.path.abspath(os.path.join(ordered_dir, saved_name)))
        dataset_path = ordered_dir

    return JsonResponse({
        "run_id": run_id,
        "dataset_path": dataset_path,
        "file_count": len(saved_paths),
        "dataset_type": "video" if video_files else "images",
    })


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
        tool["dataset_path"] = request.GET.get("dataset_path", "")
        selected_tools.append(tool)

    return render(
        request,
        "core_orchestrator/client_run_workspace.html",
        {"selected_tools": selected_tools},
    )


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
