from __future__ import annotations

import threading
from dataclasses import dataclass

from core.analysis import AnalysisCoordinator
from core.config import ConfigStore
from core.services.controller import ServiceController
from core.services.host_listener import HostListenerManager
from core.storage import state_child
from web.uploads import UploadStore


@dataclass(slots=True)
class ApplicationRuntime:
    config_store: ConfigStore
    controller: ServiceController
    host_listener: HostListenerManager
    analysis: AnalysisCoordinator
    uploads: UploadStore
    config_lock: threading.RLock


_runtime: ApplicationRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> ApplicationRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            log_dir = state_child("logs")
            controller = ServiceController(log_dir=log_dir)
            _runtime = ApplicationRuntime(
                config_store=ConfigStore(),
                controller=controller,
                host_listener=HostListenerManager(log_dir=log_dir / "instruction"),
                analysis=AnalysisCoordinator(controller),
                uploads=UploadStore(),
                config_lock=threading.RLock(),
            )
        return _runtime


def set_runtime(runtime: ApplicationRuntime | None) -> None:
    global _runtime
    with _runtime_lock:
        _runtime = runtime
