"""
settings_api.py

Django views for the persistent settings API.
"""

from __future__ import annotations

import json
import logging

from django.http import HttpRequest, JsonResponse

from ..services.settings_store import (
    _dataclass_to_dict,
    get_settings_store,
)

logger = logging.getLogger(__name__)


def _error_response(code: str, message: str, status: int = 400) -> JsonResponse:
    return JsonResponse(
        {"error": {"code": code, "message": message, "details": {}}},
        status=status,
    )


def settings_view(request: HttpRequest) -> JsonResponse:
    """
    GET  /api/settings/  -> return current normalized settings.
    PUT  /api/settings/  -> merge request body and save.
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
            return JsonResponse(_dataclass_to_dict(settings))
        except Exception as exc:
            logger.exception("Failed to save settings")
            return _error_response("settings_save_failed", f"Failed to save settings: {exc}", status=500)

    return _error_response("method_not_allowed", "Method not allowed", status=405)


def settings_reset(request: HttpRequest) -> JsonResponse:
    """POST /api/settings/reset/ -> reset to factory defaults."""
    if request.method != "POST":
        return _error_response("method_not_allowed", "Method not allowed", status=405)

    store = get_settings_store()
    settings = store.reset_to_defaults()
    return JsonResponse(_dataclass_to_dict(settings))