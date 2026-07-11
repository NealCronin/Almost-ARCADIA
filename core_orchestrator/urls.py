"""
URL routing for the core_orchestrator application.
"""

from django.urls import path

from . import views

app_name = "core_orchestrator"

urlpatterns = [
    # --- Public / Landing ---
    path("", views.landing_page, name="landing"),

    # --- Settings API ---
    path("api/settings/", views.settings_view, name="settings"),
    path("api/settings/reset/", views.settings_reset, name="settings_reset"),

    # --- Host Portal ---
    path("host/", views.host_portal, name="host_portal"),
    path("api/host/evaluate-llm/", views.host_evaluate_llm, name="host_evaluate_llm"),
    path("api/host/evaluate-sam3/", views.host_evaluate_sam3, name="host_evaluate_sam3"),
    path("api/host/status/", views.host_status, name="host_status"),

    # --- Client Portal ---
    path("client/", views.client_portal, name="client_portal"),
    path("client/heatmap/", views.heatmap_dashboard, name="heatmap_dashboard"),
    path("stream/heatmap/", views.heatmap_stream, name="heatmap_stream"),
]