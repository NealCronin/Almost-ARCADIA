from django.urls import path

from web import views

urlpatterns = [
    path("", views.home, name="home"),
    path("host/", views.nodes, name="host"),
    path("host/save/", views.save_host_listener, name="save_host_listener"),
    path("host/sam3/checkpoint/save/", views.save_host_sam3_checkpoint, name="save_host_sam3_checkpoint"),
    path("host/status/", views.host_listener_status, name="host_listener_status"),
    path("client/", views.client_portal, name="client"),
    path("client/priority-map/", views.analysis_page, name="priority_map"),
    path("client/priority-map/models/", views.services, name="priority_map_models"),
    path(
        "client/priority-map/models/llm/inspect-repository/",
        views.inspect_llm_repository,
        name="inspect_llm_repository",
    ),
    path("client/priority-map/models/llm/test-chat/", views.test_llm_chat, name="test_llm_chat"),
    path("client/priority-map/models/visual_llm/test-chat/", views.test_visual_llm_chat, name="test_visual_llm_chat"),
    path(
        "client/priority-map/models/sam3/checkpoint/upload/",
        views.upload_sam3_checkpoint,
        name="upload_sam3_checkpoint",
    ),
    path("client/priority-map/models/sam3/test/", views.test_sam3, name="test_sam3"),
    path("client/priority-map/models/nodes/add/", views.add_remote_node, name="add_remote_node"),
    path("client/priority-map/models/nodes/<str:node_name>/edit/", views.edit_remote_node, name="edit_remote_node"),
    path(
        "client/priority-map/models/nodes/<str:node_name>/delete/", views.delete_remote_node, name="delete_remote_node"
    ),
    path("client/priority-map/models/nodes/<str:node_name>/test/", views.test_remote_node, name="test_remote_node"),
    path("client/priority-map/uploads/", views.uploads, name="priority_map_uploads"),
    path("client/priority-map/uploads/<str:upload_id>/delete/", views.delete_upload, name="delete_upload"),
    path("client/priority-map/runs/", views.start_analysis, name="priority_map_runs"),
    path("client/priority-map/runs/<str:run_id>/cancel/", views.cancel_run, name="cancel_run"),
    path("client/priority-map/runs/<str:run_id>/stream/", views.run_stream, name="run_stream"),
    path("client/priority-map/runs/<str:run_id>/artifacts/", views.run_artifacts, name="run_artifacts"),
    path(
        "client/priority-map/runs/<str:run_id>/artifacts/<path:artifact_path>/", views.run_artifact, name="run_artifact"
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
