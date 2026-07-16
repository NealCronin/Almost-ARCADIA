from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import AppConfig, ConfigStore
from core.errors import ConfigurationError


def test_config_defaults_include_local_node() -> None:
    config = AppConfig()
    assert config.nodes["local"].mode == "local"
    assert config.priority_map.pipeline.sam_step == 5


def test_missing_config_copies_sibling_default(tmp_path) -> None:
    default_path = tmp_path / "default_config.json"
    payload = '{"nodes":{"local":{"mode":"local","host":"seed-host"}},"output_root":"seeded-output"}\n'
    default_path.write_text(payload, encoding="utf-8")

    config_path = tmp_path / "config.json"
    config = ConfigStore(config_path).load()

    assert config.nodes["local"].host == "seed-host"
    assert config.priority_map.output.root.name == "seeded-output"
    assert config_path.read_text(encoding="utf-8") == payload
    ConfigStore(config_path).save(config)
    saved = config_path.read_text(encoding="utf-8")
    assert '"tools"' in saved
    assert '"output_root"' not in saved


def test_legacy_configuration_migrates_without_losing_priority_map_settings(tmp_path) -> None:
    store = ConfigStore(tmp_path / "config.json")
    legacy = {
        "nodes": {"local": {"mode": "local", "host": "127.0.0.1"}},
        "services": {"llm": {"node": "local", "service_type": "llm", "port": 8081, "settings": {"n_ctx": 4096}}},
        "pipeline": {"sam_step": 9, "task": "Find roads"},
        "output_root": "saved-runs",
    }
    store.path.write_text(json.dumps(legacy), encoding="utf-8")
    config = store.load()
    assert config.priority_map.services["llm"].settings["n_ctx"] == 4096
    assert config.priority_map.pipeline.sam_step == 9
    assert config.priority_map.output.root == Path("saved-runs")
    store.save(config)
    assert store.load().to_dict() == {
        "nodes": {"local": {"mode": "local", "host": "127.0.0.1"}},
        "tools": {
            "priority-map": {
                "services": {
                    "llm": {"node": "local", "service_type": "llm", "port": 8081, "settings": {"n_ctx": 4096}}
                },
                "pipeline": config.priority_map.pipeline.to_dict(),
                "output": {"root": "saved-runs", "preview": "mjpeg"},
            }
        },
    }


def test_config_round_trip_and_atomic_save(tmp_path) -> None:
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    original = AppConfig.from_dict(
        {
            "tools": {"priority-map": {"output": {"root": "runs", "preview": "mjpeg"}}},
            "nodes": {"local": {"mode": "local", "host": "localhost"}},
        }
    )
    store.save(original)
    loaded = store.load()
    assert loaded.to_dict() == original.to_dict()
    assert not list(tmp_path.glob(".*.tmp"))


def test_malformed_config_raises(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        ConfigStore(path).load()
