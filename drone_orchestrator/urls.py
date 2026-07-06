"""
Top-level URL configuration for the drone orchestrator project.
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("core_orchestrator.urls")),
]
