"""Tests for arcadia.config — JSON configuration storage module.

Covers:
  1. Defaults — missing files, local node, mutable state isolation
  2. Round-trips — full config, nodes, nested services, pipeline settings,
     stage assignments, input/output paths
  3. Filesystem behavior — parent dirs, indentation, trailing newline,
     no auto-create, file replacement
  4. Failures — invalid JSON, path in error, non-object top-level,
     invalid nested data, failed write cleanup
"""

import json
import os
from pathlib import Path

import pytest

from arcadia.config import JsonConfigRepository
from arcadia.config.models import AppConfig, ConfigError
from arcadia.contracts import ModelSpec, NodeConfig, ServiceSpec


# ---------------------------------------------------------------------------
# Section 1 — Defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    """Default configuration behaviour."""

    def test_load_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        """Loading a non-existent file returns a default AppConfig with a local node."""
        repo = JsonConfigRepository(tmp_path / "nonexistent.json")
        config = repo.load()

        assert "local" in config.nodes
        assert config.nodes["local"].host == "127.0.0.1"
        assert config.nodes["local"].instruction_port == 9000
        assert config.nodes["local"].local is True

    def test_default_loads_return_independent_configs(self, tmp_path: Path) -> None:
        """Two loads from a missing file must return independent AppConfig instances."""
        repo = JsonConfigRepository(tmp_path / "nonexistent.json")

        first = repo.load()
        second = repo.load()

        first.nodes["local"].host = "changed"
        assert second.nodes["local"].host == "127.0.0.1"

    def test_default_configs_no_shared_mutable_state(self) -> None:
        """Two independently created AppConfig instances must not share mutable state."""
        config_a = AppConfig()
        config_b = AppConfig()

        config_a.pipeline_settings["key"] = "value"
        config_a.nodes["x"] = NodeConfig(name="x", host="0.0.0.0", instruction_port=8080, local=False)

        assert config_b.pipeline_settings == {}
        assert config_b.nodes == {}

    def test_two_loads_from_same_file_return_independent_dicts(self, tmp_path: Path) -> None:
        """Two loads from the same file must return independent dicts."""
        config = AppConfig(
            nodes={"local": NodeConfig(name="local", host="127.0.0.1", instruction_port=9000, local=True)},
            pipeline_settings={"a": 1},
        )
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config)

        d1 = repo.load().to_dict()
        d2 = repo.load().to_dict()

        d1["pipeline_settings"]["a"] = 999
        assert d2["pipeline_settings"]["a"] == 1

    def test_from_dict_returns_independent_config(self, tmp_path: Path) -> None:
        """Two loads from the same file must return independent AppConfig instances."""
        config = AppConfig(
            nodes={"local": NodeConfig(name="local", host="127.0.0.1", instruction_port=9000, local=True)},
            pipeline_settings={"a": 1},
        )
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config)

        c1 = repo.load()
        c2 = repo.load()

        c1.pipeline_settings["a"] = 999
        assert c2.pipeline_settings["a"] == 1

        c1.nodes["local"].host = "0.0.0.0"
        assert c2.nodes["local"].host == "127.0.0.1"


# ---------------------------------------------------------------------------
# Section 2 — Round-trips
# ---------------------------------------------------------------------------

class TestRoundTrips:
    """Full configuration round-trip tests."""

    def test_full_config_roundtrip(self, tmp_path: Path) -> None:
        """A complete AppConfig with all fields survives save/load."""
        scene_model = ModelSpec(repository="openai", filename="gpt-4", local_path=None)
        seg_model = ModelSpec(repository="stability", filename="sam", local_path=None)

        scene_service = ServiceSpec(
            service_type="scene",
            port=8000,
            model=scene_model,
            settings={"temperature": 0.7},
        )
        seg_service = ServiceSpec(
            service_type="segmentation",
            port=8001,
            model=seg_model,
            settings={"threshold": 0.5},
        )

        config = AppConfig(
            nodes={
                "scene": NodeConfig(name="scene", host="10.0.0.1", instruction_port=8000, local=False),
                "segmentation": NodeConfig(name="segmentation", host="10.0.0.2", instruction_port=8001, local=False),
            },
            instruction_host="127.0.0.1",
            instruction_port=9000,
            scene_service=scene_service,
            segmentation_service=seg_service,
            scene_node="scene",
            segmentation_node="segmentation",
            input_path="/data/input",
            output_path="/data/output",
            pipeline_settings={"sam_step": 5, "max_retries": 3},
        )

        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config)
        restored = repo.load()

        assert restored.nodes["scene"].name == "scene"
        assert restored.nodes["scene"].host == "10.0.0.1"
        assert restored.nodes["scene"].instruction_port == 8000
        assert restored.nodes["scene"].local is False

        assert restored.nodes["segmentation"].name == "segmentation"
        assert restored.nodes["segmentation"].host == "10.0.0.2"
        assert restored.nodes["segmentation"].instruction_port == 8001
        assert restored.nodes["segmentation"].local is False

        assert restored.scene_service.service_type == "scene"
        assert restored.scene_service.port == 8000
        assert restored.scene_service.model.repository == "openai"
        assert restored.scene_service.model.filename == "gpt-4"
        assert restored.scene_service.settings["temperature"] == 0.7

        assert restored.segmentation_service.service_type == "segmentation"
        assert restored.segmentation_service.port == 8001
        assert restored.segmentation_service.model.repository == "stability"
        assert restored.segmentation_service.model.filename == "sam"
        assert restored.segmentation_service.settings["threshold"] == 0.5

        assert restored.scene_node == "scene"
        assert restored.segmentation_node == "segmentation"
        assert restored.input_path == "/data/input"
        assert restored.output_path == "/data/output"
        assert restored.pipeline_settings["sam_step"] == 5
        assert restored.pipeline_settings["max_retries"] == 3

    def test_named_nodes_roundtrip(self, tmp_path: Path) -> None:
        """Multiple named nodes with different configs survive round-trip."""
        nodes = {
            "node1": NodeConfig(name="node1", host="10.0.0.1", instruction_port=8001, local=False),
            "node2": NodeConfig(name="node2", host="10.0.0.2", instruction_port=8002, local=False),
            "node3": NodeConfig(name="node3", host="10.0.0.3", instruction_port=8003, local=False),
        }
        config = AppConfig(nodes=nodes)

        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config)
        restored = repo.load()

        assert len(restored.nodes) == 3
        for name in nodes:
            assert name in restored.nodes
            assert restored.nodes[name].host == nodes[name].host
            assert restored.nodes[name].instruction_port == nodes[name].instruction_port

    def test_nested_service_spec_roundtrip(self, tmp_path: Path) -> None:
        """ServiceSpec with nested ModelSpec survives round-trip."""
        model = ModelSpec(repository="huggingface", filename="llama-2-7b", local_path="/models/llama-2-7b")
        service = ServiceSpec(service_type="generation", port=9000, model=model, settings={"max_tokens": 2048})
        config = AppConfig(scene_service=service)

        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config)
        restored = repo.load()

        assert restored.scene_service is not None
        assert restored.scene_service.service_type == "generation"
        assert restored.scene_service.port == 9000
        assert restored.scene_service.model.repository == "huggingface"
        assert restored.scene_service.model.filename == "llama-2-7b"
        assert restored.scene_service.model.local_path == "/models/llama-2-7b"
        assert restored.scene_service.settings["max_tokens"] == 2048

    def test_pipeline_settings_roundtrip(self, tmp_path: Path) -> None:
        """Arbitrary pipeline settings survive round-trip."""
        settings = {
            "sam_step": 5,
            "max_retries": 3,
            "timeout": 300,
            "verbose": True,
            "nested": {"key": "value", "list": [1, 2, 3]},
        }
        config = AppConfig(pipeline_settings=settings)

        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config)
        restored = repo.load()

        assert restored.pipeline_settings == settings

    def test_stage_node_assignments_roundtrip(self, tmp_path: Path) -> None:
        """Scene and segmentation node assignments survive round-trip."""
        config = AppConfig(
            nodes={"scene": NodeConfig(name="scene", host="10.0.0.1", instruction_port=8000, local=False)},
            scene_node="scene",
            segmentation_node="segmentation",
        )

        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config)
        restored = repo.load()

        assert restored.scene_node == "scene"
        assert restored.segmentation_node == "segmentation"

    def test_input_output_paths_roundtrip(self, tmp_path: Path) -> None:
        """Non-empty input/output paths survive round-trip."""
        config = AppConfig(input_path="/data/input", output_path="/data/output")

        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config)
        restored = repo.load()

        assert restored.input_path == "/data/input"
        assert restored.output_path == "/data/output"


# ---------------------------------------------------------------------------
# Section 3 — Filesystem behavior
# ---------------------------------------------------------------------------

class TestFilesystem:
    """Filesystem-level behaviour of save/load."""

    def test_parent_directories_created(self, tmp_path: Path) -> None:
        """Saving to a nested non-existent path creates parent directories."""
        target = tmp_path / "nested" / "deep" / "config.json"
        repo = JsonConfigRepository(target)
        repo.save(AppConfig())

        assert target.exists()
        assert target.parent.exists()

    def test_saved_json_is_indented(self, tmp_path: Path) -> None:
        """Saved JSON uses indentation (indent=2 produces newlines)."""
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(AppConfig())

        content = repo.path.read_text()
        assert "\n" in content

    def test_saved_json_ends_with_newline(self, tmp_path: Path) -> None:
        """Saved JSON ends with a trailing newline."""
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(AppConfig())

        content = repo.path.read_text()
        assert content.endswith("\n")

    def test_load_does_not_create_missing_file(self, tmp_path: Path) -> None:
        """Loading a non-existent file does not create it."""
        repo = JsonConfigRepository(tmp_path / "nonexistent.json")
        repo.load()

        assert not (tmp_path / "nonexistent.json").exists()

    def test_existing_file_replaced(self, tmp_path: Path) -> None:
        """Saving a new config overwrites the existing file."""
        config1 = AppConfig(input_path="/old")
        config2 = AppConfig(input_path="/new")

        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(config1)
        repo.save(config2)

        restored = repo.load()
        assert restored.input_path == "/new"


# ---------------------------------------------------------------------------
# Section 4 — Failures
# ---------------------------------------------------------------------------

class TestFailures:
    """Error handling for invalid or failed operations."""

    def test_invalid_json_raises_config_error(self, tmp_path: Path) -> None:
        """Invalid JSON in the file raises ConfigError."""
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.path.write_text("{ invalid json }")

        with pytest.raises(ConfigError) as exc_info:
            repo.load()

        assert "Invalid JSON" in str(exc_info.value)

    def test_config_error_includes_path(self, tmp_path: Path) -> None:
        """ConfigError message includes the file path."""
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.path.write_text("{ invalid json }")

        with pytest.raises(ConfigError) as exc_info:
            repo.load()

        assert str(tmp_path / "config.json") in str(exc_info.value)

    def test_non_object_top_level_fails_clearly(self, tmp_path: Path) -> None:
        """A JSON array at the top level raises ConfigError."""
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.path.write_text("[1, 2, 3]")

        with pytest.raises(ConfigError) as exc_info:
            repo.load()

        assert "top-level must be a JSON object" in str(exc_info.value)

    def test_invalid_nested_node_data_fails_clearly(self, tmp_path: Path) -> None:
        """A node missing required fields raises ConfigError."""
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.path.write_text(json.dumps({
            "nodes": {"bad": {"host": "10.0.0.1", "instruction_port": 8000}},
        }))

        with pytest.raises(ConfigError) as exc_info:
            repo.load()

        assert "Invalid node" in str(exc_info.value)

    def test_invalid_nested_service_data_fails_clearly(self, tmp_path: Path) -> None:
        """A service missing service_type raises ConfigError."""
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.path.write_text(json.dumps({
            "scene_service": {"port": 8000, "model": {"repository": "x", "filename": "y"}},
        }))

        with pytest.raises(ConfigError) as exc_info:
            repo.load()

        assert "Invalid scene_service" in str(exc_info.value)

    def test_failed_write_leaves_existing_config_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If os.replace fails, the existing config file is not corrupted."""
        original = AppConfig(input_path="/original")
        repo = JsonConfigRepository(tmp_path / "config.json")
        repo.save(original)

        monkeypatch.setattr("os.replace", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("simulated failure")))

        new_config = AppConfig(input_path="/new")
        with pytest.raises(ConfigError):
            repo.save(new_config)

        # Original config must still be intact
        restored = repo.load()
        assert restored.input_path == "/original"

    def test_temp_file_removed_after_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """If os.replace fails, the temp file is cleaned up."""
        repo = JsonConfigRepository(tmp_path / "config.json")

        monkeypatch.setattr("os.replace", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("simulated failure")))

        with pytest.raises(ConfigError):
            repo.save(AppConfig())

        # No temp files should remain
        tmp_files = list(tmp_path.glob(".arcadia-config-*.tmp"))
        assert len(tmp_files) == 0, f"Temp files remain: {tmp_files}"
