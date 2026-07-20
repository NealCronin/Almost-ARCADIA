from __future__ import annotations

import os
import sys

from django.apps import AppConfig


class WebConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web"

    def ready(self) -> None:
        """Start one owned instruction listener only in the real runserver process."""
        if os.environ.get("ARCADIA_DISABLE_AUTO_LISTENER") == "1":
            return
        if "runserver" not in sys.argv:
            return
        autoreload_child = os.environ.get("RUN_MAIN") == "true"
        no_reload = "--noreload" in sys.argv
        if not (autoreload_child or no_reload):
            return
        from web.runtime import get_runtime

        runtime = get_runtime()
        if runtime.host_listener.status().running:
            return
        try:
            runtime.host_listener.start(runtime.config_store.load().host_listener)
        except Exception:
            # Django remains available so the Host page can show and repair the failure.
            return
