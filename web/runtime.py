from __future__ import annotations

import atexit
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock

from django.conf import settings

from core.analysis import AnalysisCoordinator
from core.config import ConfigStore
from core.services.controller import ServiceController
from core.services.host_listener import HostListenerController, HostListenerError
from web.tools import TOOLS
from web.uploads import UploadStore


@dataclass(slots=True)
class ApplicationRuntime:
    config_store: ConfigStore
    host_listener: HostListenerController
    controller: ServiceController
    analysis: AnalysisCoordinator
    uploads: UploadStore
    config_lock: RLock = field(default_factory=RLock, repr=False)


_runtime: ApplicationRuntime | None = None
_lock = Lock()


def should_autostart_host_listener(argv: list[str] | None = None, environ: dict[str, str] | None = None) -> bool:
    command = (argv or sys.argv)[1:]
    environment = environ or os.environ
    if not command or command[0] != "runserver":
        return False
    if "RUN_MAIN" in environment:
        return environment["RUN_MAIN"].lower() == "true"
    return "--noreload" in command


def close_runtime() -> None:
    global _runtime
    with _lock:
        if _runtime is None:
            return
        _runtime.host_listener.close()
        _runtime.controller.stop_all()
        _runtime = None


atexit.register(close_runtime)


def get_runtime() -> ApplicationRuntime:
    global _runtime
    with _lock:
        if _runtime is None:
            base_dir = Path(getattr(settings, "BASE_DIR", Path.cwd()))
            config_path = Path(os.environ.get("ARCADIA_CONFIG", base_dir / "config.json"))
            log_dir = Path(os.environ.get("ARCADIA_LOG_DIR", base_dir / "logs"))
            store = ConfigStore(config_path)
            config = store.load()
            host_listener = HostListenerController(log_dir=log_dir / "instruction")
            if should_autostart_host_listener():
                try:
                    host_listener.ensure_started(config.host_listener)
                except HostListenerError:
                    pass
            controller = ServiceController(public_host=config.host_listener.host, log_dir=log_dir)
            _runtime = ApplicationRuntime(
                store,
                host_listener,
                controller,
                AnalysisCoordinator(store, controller, TOOLS["priority-map"].runner_factory()),
                UploadStore(base_dir / "workspace" / "uploads"),
            )
        return _runtime
