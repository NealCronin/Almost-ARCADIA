from __future__ import annotations

import pytest

from core.config import AppConfig, ConfigStore
from core.errors import ConfigurationError


def test_config_defaults_include_local_node() -> None:
    config = AppConfig()
    assert config.nodes["local"].mode == "local"
    assert config.pipeline.sam_step == 5



def test_missing_config_copies_sibling_default(tmp_path) -> None:
    default_path = tmp_path / "default_config.json"
    payload = '{"nodes":{"local":{"mode":"local","host":"seed-host"}},"output_root":"seeded-output"}\n'
    default_path.write_text(payload, encoding="utf-8")

    config_path = tmp_path / "config.json"
    config = ConfigStore(config_path).load()

    assert config.nodes["local"].host == "seed-host"
    assert config.output_root.name == "seeded-output"
    assert config_path.read_text(encoding="utf-8") == payload

def test_config_round_trip_and_atomic_save(tmp_path) -> None:
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    original = AppConfig.from_dict({"output_root": "runs", "nodes": {"local": {"mode": "local", "host": "localhost"}}})
    store.save(original)
    loaded = store.load()
    assert loaded.to_dict() == original.to_dict()
    assert not list(tmp_path.glob(".*.tmp"))


def test_malformed_config_raises(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        ConfigStore(path).load()
