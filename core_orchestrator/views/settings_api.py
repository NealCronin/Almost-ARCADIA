"""
settings_api.py

Django views for the persistent settings API.
CSRF-exempt for trusted local application use — documented and intentional.
"""

from __future__ import annotations

import json
import logging

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ..services.settings_store import (
    SettingsValidationError,
    _dataclass_to_dict,
    get_settings_store,
)

logger = logging.getLogger(__name__)


def _error_response(code: str, message: str, status: int = 400, details: dict | None = None) -> JsonResponse:
    return JsonResponse(
        {"error": {"code": code, "message": message, "details": details or {}}},
        status=status,
    )


@csrf_exempt
def settings_view(request: HttpRequest) -> JsonResponse:
    """
    GET  /api/settings/  -> return current normalized settings.
    PUT  /api/settings/  -> merge request body and save.

    CSRF is explicitly exempted for this trusted local application API.
    The application is designed for localhost/LAN/VPN use, not public
    internet deployment.
    """
    store = get_settings_store()

    if request.method == "GET":
        settings = store.load()
        return JsonResponse(_dataclass_to_dict(settings))

    if request.method == "PUT":
        try:
            body = json.loads(request.body)
        except json.JSONDecodeError:
            return _error_response("invalid_json", "Request body is not valid JSON")

        if not isinstance(body, dict):
            return _error_response("invalid_json", "Request body must be a JSON object")

        try:
            settings = store.update(body)
        except SettingsValidationError as exc:
            return _error_response(
                "invalid_configuration",
                "The submitted settings are invalid.",
                details={exc.field_path: str(exc)},
            )
        except Exception as exc:
            logger.exception("Failed to save settings")
            return _error_response("settings_save_failed", "Failed to save settings", status=500)

        return JsonResponse(_dataclass_to_dict(settings))

    return _error_response("method_not_allowed", "Method not allowed", status=405)


@csrf_exempt
def settings_reset(request: HttpRequest) -> JsonResponse:
    """POST /api/settings/reset/ -> reset to factory defaults."""
    if request.method != "POST":
        return _error_response("method_not_allowed", "Method not allowed", status=405)

    store = get_settings_store()
    settings = store.reset_to_defaults()
    return JsonResponse(_dataclass_to_dict(settings))


@csrf_exempt
def command_preview(request: HttpRequest) -> JsonResponse:
    """
    POST /api/services/command-preview/ -> return generated command array.

    Payload::

        {"service_id": "host:llm", "configuration": {...}}
    """
    if request.method != "POST":
        return _error_response("method_not_allowed", "Method not allowed", status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response("invalid_json", "Invalid JSON")

    service_id = body.get("service_id", "")
    config = body.get("configuration", {})

    if service_id.endswith(":llm"):
        from ..services.settings_store import LLMServiceSettings
        from ..services.command_builder import build_llm_command, preview_command_parts

        try:
            settings = LLMServiceSettings(**config)
        except TypeError as exc:
            return _error_response("invalid_configuration", str(exc))

        parts = preview_command_parts(settings)
        warnings = _detect_duplicate_flags(parts.get("argument_array", []))
        return JsonResponse({**parts, "warnings": warnings})

    elif service_id.endswith(":sam3"):
        # SAM runs in-process; no subprocess command
        return JsonResponse({
            "argument_array": [],
            "display_command": "Managed SAM runs in-process using the configured weights path.",
            "structured_arguments": [],
            "raw_arguments": [],
            "warnings": [],
        })

    return _error_response("invalid_service_id", f"Unknown service: {service_id}")


@csrf_exempt
def service_logs(request: HttpRequest) -> JsonResponse:
    """
    GET /api/services/logs/?service_id=host:llm&tail=200 -> recent log lines.
    """
    if request.method != "GET":
        return _error_response("method_not_allowed", "Method not allowed", status=405)

    service_id = request.GET.get("service_id", "")
    tail = int(request.GET.get("tail", 200))

    if not service_id:
        return _error_response("missing_param", "service_id is required")

    from ..services.service_manager import get_service_manager
    sm = get_service_manager()

    # Get ServiceManager logs
    sm_logs = sm.get_logs(service_id, tail=tail)

    # Also get ProcessManager output if available
    process_stdout = []
    process_stderr = []
    try:
        from ..utils.model_host.process_manager import ProcessManager
        pm = ProcessManager.instance()
        stdout, stderr = pm.get_output(service_id)
        process_stdout = stdout[-tail:]
        process_stderr = stderr[-tail:]
    except Exception:
        pass

    lines = []
    for line in sm_logs:
        lines.append({"stream": "service", "text": line})
    for line in process_stdout:
        lines.append({"stream": "stdout", "text": line})
    for line in process_stderr:
        lines.append({"stream": "stderr", "text": line})

    return JsonResponse({"service_id": service_id, "lines": lines[-tail:]})


@csrf_exempt
def test_host_connection(request: HttpRequest) -> JsonResponse:
    """
    POST /api/client/test-host/ -> test connection to a remote Host.

    Payload::

        {"host": "100.96.40.81", "port": 8080, "scheme": "http"}

    This is a Django-side proxy to avoid browser CORS issues when the
    Host listener is on a different origin.
    """
    if request.method != "POST":
        return _error_response("method_not_allowed", "Method not allowed", status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return _error_response("invalid_json", "Invalid JSON")

    host = body.get("host", "127.0.0.1")
    port = body.get("port", 8080)
    scheme = body.get("scheme", "http")

    try:
        port = int(port)
        if port < 1 or port > 65535:
            raise ValueError("port out of range")
    except (TypeError, ValueError):
        return _error_response("invalid_configuration", "Invalid port number")

    from ..utils.model_host.remote_client_helper import RemoteClientHelper, RemoteClientError

    client = RemoteClientHelper(base_url=f"{scheme}://{host}:{port}", timeout=5)
    try:
        status = client.get_status()
        return JsonResponse({"connected": True, "status": status})
    except RemoteClientError as exc:
        return JsonResponse({"connected": False, "error": {"code": "remote_host_unreachable", "message": str(exc)}})


def _detect_duplicate_flags(cmd: list[str]) -> list[str]:
    """Detect duplicate flags in a command array."""
    warnings = []
    seen_flags = set()
    for i, token in enumerate(cmd):
        if token.startswith("--") and i < len(cmd) - 1:
            if token in seen_flags:
                warnings.append(f"Duplicate flag detected: {token}")
            seen_flags.add(token)
    return warnings