from django.urls import path

from web import views

urlpatterns = [
    path("", views.home, name="home"),
    path("host/", views.nodes, name="host"),
    path("host/save/", views.save_host_listener, name="save_host_listener"),
    path("host/status/", views.host_listener_status, name="host_listener_status"),
    path("client/", views.client_portal, name="client"),
    path("client/priority-map/", views.analysis_page, name="priority_map"),
    path("client/priority-map/models/", views.services, name="priority_map_models"),
    path("client/priority-map/uploads/", views.uploads, name="priority_map_uploads"),
    path("client/priority-map/uploads/<str:upload_id>/delete/", views.delete_upload, name="delete_upload"),
    path("client/priority-map/runs/", views.start_analysis, name="priority_map_runs"),
    path("client/priority-map/runs/<str:run_id>/cancel/", views.cancel_run, name="cancel_run"),
    path("client/priority-map/runs/<str:run_id>/stream/", views.run_stream, name="run_stream"),
    path("client/priority-map/runs/<str:run_id>/artifacts/", views.run_artifacts, name="run_artifacts"),
    path(
        "client/priority-map/runs/<str:run_id>/artifacts/<path:artifact_path>/",
        views.run_artifact,
        name="run_artifact",
    ),
    path("services/", views.services, name="services"),
    path("services/<str:service_name>/start/", views.start_service, name="start_service"),
    path("services/<str:service_name>/stop/", views.stop_service, name="stop_service"),
    path("services/logs/<int:port>/", views.service_logs, name="service_logs"),
    path("analysis/", views.analysis_page, name="analysis"),
    path("analysis/configure/", views.save_pipeline, name="save_pipeline"),
    path("analysis/start/", views.start_analysis, name="start_analysis"),
    path("analysis/status/", views.analysis_status, name="analysis_status"),
    path("results/", views.results, name="results"),
    path("endpoint-test/", views.endpoint_test, name="endpoint_test"),
    path("endpoint-test/run/", views.run_endpoint_test, name="run_endpoint_test"),
]
