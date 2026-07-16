from __future__ import annotations

from django.apps import AppConfig


class WebConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web"

    def ready(self) -> None:
        from web.runtime import get_runtime, should_autostart_host_listener

        if should_autostart_host_listener():
            get_runtime()
