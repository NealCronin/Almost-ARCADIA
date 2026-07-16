from django.urls import path

from web import views

urlpatterns = [
    path("", views.home, name="home"),
    path("services/", views.services, name="services"),
    path("services/<str:service_name>/start/", views.start_service, name="start_service"),
    path("services/<str:service_name>/stop/", views.stop_service, name="stop_service"),
    path("services/logs/<int:port>/", views.service_logs, name="service_logs"),
    path("nodes/", views.nodes, name="nodes"),
    path("nodes/<str:node_name>/save/", views.save_node, name="save_node"),
    path("analysis/", views.analysis_page, name="analysis"),
    path("analysis/configure/", views.save_pipeline, name="save_pipeline"),
    path("analysis/start/", views.start_analysis, name="start_analysis"),
    path("analysis/status/", views.analysis_status, name="analysis_status"),
    path("results/", views.results, name="results"),
    path("endpoint-test/", views.endpoint_test, name="endpoint_test"),
    path("endpoint-test/run/", views.run_endpoint_test, name="run_endpoint_test"),
]
