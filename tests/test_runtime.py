from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import Mock

from core.config import AppConfig, ConfigStore
from core.services.host_listener import HostListenerError
from web import runtime as runtime_module
from web.apps import WebConfig


def test_autostart_guard_allows_only_runserver_child_or_noreload() -> None:
    assert not runtime_module.should_autostart_host_listener(["manage.py", "check"], {})
    assert not runtime_module.should_autostart_host_listener(["manage.py", "migrate"], {})
    assert not runtime_module.should_autostart_host_listener(["manage.py", "runserver"], {})
    assert runtime_module.should_autostart_host_listener(["manage.py", "runserver"], {"RUN_MAIN": "true"})
    assert not runtime_module.should_autostart_host_listener(["manage.py", "runserver"], {"RUN_MAIN": "false"})
    assert runtime_module.should_autostart_host_listener(["manage.py", "runserver", "--noreload"], {})


def test_web_app_ready_starts_only_when_runtime_guard_allows(monkeypatch) -> None:
    start = Mock()
    config = WebConfig("web", importlib.import_module("web"))
    monkeypatch.setattr(runtime_module, "get_runtime", start)
    monkeypatch.setattr(runtime_module, "should_autostart_host_listener", lambda: False)

    config.ready()

    start.assert_not_called()
    monkeypatch.setattr(runtime_module, "should_autostart_host_listener", lambda: True)
    config.ready()
    start.assert_called_once()


def test_runtime_owns_one_listener_controller(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config = AppConfig()
    config.host_listener.host = "127.0.0.1"
    config.host_listener.port = 9010
    ConfigStore(config_path).save(config)
    created = []

    class FakeHostListener:
        def __init__(self, *, log_dir):
            self.log_dir = log_dir
            self.started = []
            self.closed = False
            created.append(self)

        def ensure_started(self, listener_config):
            self.started.append(listener_config)

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime_module, "HostListenerController", FakeHostListener)
    monkeypatch.setattr(runtime_module, "should_autostart_host_listener", lambda: True)
    monkeypatch.setattr(runtime_module.settings, "BASE_DIR", tmp_path)
    monkeypatch.setenv("ARCADIA_CONFIG", str(config_path))
    monkeypatch.setenv("ARCADIA_LOG_DIR", str(tmp_path / "logs"))
    runtime_module.close_runtime()

    first = runtime_module.get_runtime()
    second = runtime_module.get_runtime()

    assert first is second
    assert first.controller.public_host == "127.0.0.1"
    assert created[0].log_dir == tmp_path / "logs" / "instruction"
    assert created[0].started == [config.host_listener]
    runtime_module.close_runtime()
    assert created[0].closed


def test_runtime_keeps_django_available_when_listener_startup_fails(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    ConfigStore(config_path).save(AppConfig())

    class FailingHostListener:
        def __init__(self, *, log_dir):
            self.log_dir = log_dir
            self.closed = False

        def ensure_started(self, listener_config):
            raise HostListenerError("port is already in use")

        def close(self):
            self.closed = True

    monkeypatch.setattr(runtime_module, "HostListenerController", FailingHostListener)
    monkeypatch.setattr(runtime_module, "should_autostart_host_listener", lambda: True)
    monkeypatch.setattr(runtime_module.settings, "BASE_DIR", tmp_path)
    monkeypatch.setenv("ARCADIA_CONFIG", str(config_path))
    runtime_module.close_runtime()

    runtime = runtime_module.get_runtime()

    assert runtime.host_listener.log_dir == tmp_path / "logs" / "instruction"
    runtime_module.close_runtime()
