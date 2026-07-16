from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from django.conf import settings

from core.analysis import AnalysisCoordinator
from core.config import ConfigStore
from core.services.controller import ServiceController
from web.tools import TOOLS
from web.uploads import UploadStore


@dataclass(slots=True)
class ApplicationRuntime:
    config_store: ConfigStore
    controller: ServiceController
    analysis: AnalysisCoordinator
    uploads: UploadStore


_runtime: ApplicationRuntime | None = None
_lock = Lock()


def get_runtime() -> ApplicationRuntime:
    global _runtime
    with _lock:
        if _runtime is None:
            base_dir = Path(getattr(settings, "BASE_DIR", Path.cwd()))
            config_path = Path(os.environ.get("ARCADIA_CONFIG", base_dir / "config.json"))
            log_dir = Path(os.environ.get("ARCADIA_LOG_DIR", base_dir / "logs"))
            store = ConfigStore(config_path)
            controller = ServiceController(public_host="127.0.0.1", log_dir=log_dir)
            _runtime = ApplicationRuntime(
                store,
                controller,
                AnalysisCoordinator(store, controller, TOOLS["priority-map"].runner_factory()),
                UploadStore(base_dir / "workspace" / "uploads"),
            )
        return _runtime
