"""
core_orchestrator/views.py

Legacy entry point — re-exports all views from the refactored views package
for backward compatibility.

All new code lives in ``core_orchestrator/views/pages.py`` and
``core_orchestrator/views/settings_api.py``.
"""

from .views.pages import (
    landing_page,
    host_portal,
    host_evaluate_llm,
    host_evaluate_sam3,
    host_status,
    client_portal,
    client_run_workspace,
    heatmap_dashboard,
    heatmap_stream,
    log_request,
    get_request_logs,
    validate_sam3_model,
    validate_llm_model,
    validate_opencv,
    HostAPIHandler,
)
from .views.settings_api import settings_view, settings_reset

__all__ = [
    "landing_page",
    "host_portal",
    "host_evaluate_llm",
    "host_evaluate_sam3",
    "host_status",
    "client_portal",
    "client_run_workspace",
    "heatmap_dashboard",
    "heatmap_stream",
    "settings_view",
    "settings_reset",
    "log_request",
    "get_request_logs",
    "validate_sam3_model",
    "validate_llm_model",
    "validate_opencv",
    "HostAPIHandler",
]