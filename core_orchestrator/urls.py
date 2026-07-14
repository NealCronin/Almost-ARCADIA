"""
URL routing for the core_orchestrator application.
"""

from django.urls import path

from . import views

app_name = "core_orchestrator"

urlpatterns = [
    # --- Public / Landing ---
    path("", views.landing_page, name="landing"),

    # --- Host Portal ---
    path("host/", views.host_portal, name="host_portal"),
    path("api/host/evaluate-llm/", views.host_evaluate_llm, name="host_evaluate_llm"),
    path("api/host/evaluate-sam3/", views.host_evaluate_sam3, name="host_evaluate_sam3"),
    path("api/host/status/", views.host_status, name="host_status"),

    # --- Client Portal ---
    path("client/", views.client_portal, name="client_portal"),
    path("client/run/", views.client_run_workspace, name="client_run_workspace"),
    path("client/heatmap/", views.heatmap_dashboard, name="heatmap_dashboard"),
    path("stream/heatmap/", views.heatmap_stream, name="heatmap_stream"),
    path("api/reconstruction/upload/", views.reconstruction_upload_dataset, name="reconstruction_upload_dataset"),
    path("api/reconstruction/start/", views.reconstruction_start, name="reconstruction_start"),
    path("api/reconstruction/stop/", views.reconstruction_stop, name="reconstruction_stop"),
    path("api/reconstruction/status/", views.reconstruction_status, name="reconstruction_status"),
    path("api/reconstruction/viewer/<str:run_id>/", views.reconstruction_viewer, name="reconstruction_viewer"),
    path("api/reconstruction/artifact/<str:run_id>/<str:filename>", views.reconstruction_artifact, name="reconstruction_artifact"),
]
