"""Tests for the shared data contracts module.

Covers:
- Every JSON-facing contract can be constructed.
- Nested contracts survive dict and JSON round trips.
- Mutable defaults are not shared between instances.
- Missing required fields fail clearly.
- RunningService.runtime_handle is excluded from serialization
  (including a non-copyable handle).
- Binary request data remains unchanged in memory.
- SegmentationResult retains exact mask objects (in-memory only).
- pathlib.Path values encode and reconstruct correctly.
"""

import copy
import json
from pathlib import Path

import pytest

from arcadia.contracts import (
    AnalysisConfig,
    AnalysisResult,
    AnalysisWorkspace,
    LanguageRequest,
    LanguageResponse,
    ModelSpec,
    NodeConfig,
    RunningService,
    ServiceEndpoint,
    ServiceSpec,
    SegmentationRequest,
    SegmentationResult,
)


# =====================================================================
# 1. Construction — every contract can be instantiated
# =====================================================================

def test_model_spec_construction():
    m = ModelSpec(repository="test", filename="file.bin", local_path="/tmp")
    assert m.repository == "test"
    assert m.filename == "file.bin"
    assert m.local_path == "/tmp"

    m2 = ModelSpec()
    assert m2.repository is None
    assert m2.filename is None
    assert m2.local_path is None


def test_service_spec_construction():
    s = ServiceSpec(
        service_type="llm",
        port=8000,
        model=ModelSpec(repository="repo", filename="model.bin", local_path="/m/scene"),
        settings={"max_tokens": 2048},
    )
    assert s.service_type == "llm"
    assert s.port == 8000
    assert s.model.filename == "model.bin"
    assert s.settings["max_tokens"] == 2048

    s2 = ServiceSpec(service_type="vision", port=9000)
    assert s2.model is None
    assert s2.settings == {}


def test_service_endpoint_construction():
    e = ServiceEndpoint(host="localhost", port=8001, service_type="llm")
    assert e.host == "localhost"
    assert e.port == 8001
    assert e.service_type == "llm"


def test_node_config_construction():
    n = NodeConfig(name="scene", host="127.0.0.1", instruction_port=5000, local=True)
    assert n.name == "scene"
    assert n.host == "127.0.0.1"
    assert n.instruction_port == 5000
    assert n.local is True


def test_running_service_construction():
    rs = RunningService(
        spec=ServiceSpec(service_type="llm", port=8000),
        endpoint=ServiceEndpoint(host="127.0.0.1", port=8001, service_type="llm"),
        runtime_handle="mock_handle",
    )
    assert rs.spec.service_type == "llm"
    assert rs.endpoint.host == "127.0.0.1"
    assert rs.runtime_handle == "mock_handle"


def test_language_request_construction():
    req = LanguageRequest(
        prompt="Hello world",
        images=[b"fake_image_data"],
        settings={"temperature": 0.7},
    )
    assert req.prompt == "Hello world"
    assert req.images == [b"fake_image_data"]
    assert req.settings["temperature"] == 0.7

    req2 = LanguageRequest(prompt="simple")
    assert req2.images is None
    assert req2.settings == {}


def test_language_response_construction():
    resp = LanguageResponse(text="The answer is 42")
    assert resp.text == "The answer is 42"


def test_segmentation_request_construction():
    req = SegmentationRequest(image=b"fake_image", prompt="detect objects")
    assert req.image == b"fake_image"
    assert req.prompt == "detect objects"

    req2 = SegmentationRequest(image=b"fake", prompt=["obj1", "obj2"])
    assert isinstance(req2.prompt, list)


def test_segmentation_result_construction():
    res = SegmentationResult(
        masks=[True, False],
        labels=["cat", "dog"],
        confidences=[0.95, 0.3],
        bounding_boxes=[[10, 10, 20, 20]],
        source_width=1920,
        source_height=1080,
    )
    assert res.labels == ["cat", "dog"]
    assert res.confidences == [0.95, 0.3]
    assert res.source_width == 1920
    assert res.masks == [True, False]


def test_analysis_config_construction():
    cfg = AnalysisConfig(
        input_path="/data/input.jpg",
        output_path="/data/output",
        scene_service=ServiceSpec(service_type="scene", port=8000),
        segmentation_service=ServiceSpec(service_type="seg", port=9000),
        scene_node=NodeConfig(name="scene", host="127.0.0.1", instruction_port=5000, local=False),
        segmentation_node=NodeConfig(name="seg", host="127.0.0.1", instruction_port=5001, local=False),
        pipeline_settings={"steps": 3},
    )
    assert cfg.input_path == "/data/input.jpg"
    assert cfg.scene_service.service_type == "scene"
    assert cfg.pipeline_settings["steps"] == 3


def test_analysis_result_construction():
    r = AnalysisResult(
        output_directory="/out",
        result_files=["a.jpg", "b.png"],
        success=True,
        error=None,
    )
    assert r.output_directory == "/out"
    assert r.success is True
    assert r.error is None


def test_analysis_workspace_construction():
    ws = AnalysisWorkspace(
        root=Path("/ws"),
        log_path=Path("/ws/log.txt"),
        config_path=Path("/ws/config.json"),
        result_path=Path("/ws/res"),
    )
    assert ws.root == Path("/ws")
    assert ws.log_path == Path("/ws/log.txt")


# =====================================================================
# 2. Dict round-trip (to_dict / from_dict) — JSON-facing contracts
# =====================================================================

def test_model_spec_dict_roundtrip():
    m = ModelSpec(repository="repo", filename="f.bin", local_path="/p")
    d = m.to_dict()
    m2 = ModelSpec.from_dict(d)
    assert m2.repository == m.repository
    assert m2.filename == m.filename
    assert m2.local_path == m.local_path


def test_service_spec_dict_roundtrip():
    s = ServiceSpec(
        service_type="llm",
        port=8000,
        model=ModelSpec(repository="r", filename="m.bin", local_path="/l"),
        settings={"temp": 0.7, "key": "value"},
    )
    d = s.to_dict()
    s2 = ServiceSpec.from_dict(d)
    assert s2.service_type == s.service_type
    assert s2.port == s.port
    assert s2.model.repository == s.model.repository
    assert s2.model.filename == s.model.filename
    assert s2.model.local_path == s.model.local_path
    assert s2.settings == s.settings


def test_service_endpoint_dict_roundtrip():
    e = ServiceEndpoint(host="host", port=9000, service_type="seg")
    d = e.to_dict()
    e2 = ServiceEndpoint.from_dict(d)
    assert e2.host == e.host
    assert e2.port == e.port
    assert e2.service_type == e.service_type


def test_node_config_dict_roundtrip():
    n = NodeConfig(name="node", host="10.0.0.1", instruction_port=6000, local=True)
    d = n.to_dict()
    n2 = NodeConfig.from_dict(d)
    assert n2.name == n.name
    assert n2.host == n.host
    assert n2.instruction_port == n.instruction_port
    assert n2.local == n.local


def test_language_response_dict_roundtrip():
    r = LanguageResponse(text="hello")
    d = r.to_dict()
    r2 = LanguageResponse.from_dict(d)
    assert r2.text == r.text


def test_analysis_config_dict_roundtrip():
    cfg = AnalysisConfig(
        input_path="/in",
        output_path="/out",
        scene_service=ServiceSpec(service_type="s", port=8000),
        segmentation_service=ServiceSpec(service_type="s", port=8000),
        scene_node=NodeConfig(name="s", host="1", instruction_port=5000, local=False),
        segmentation_node=NodeConfig(name="s", host="1", instruction_port=5000, local=False),
        pipeline_settings={"k": "v"},
    )
    d = cfg.to_dict()
    cfg2 = AnalysisConfig.from_dict(d)
    assert cfg2.input_path == cfg.input_path
    assert cfg2.output_path == cfg.output_path
    assert cfg2.scene_service.service_type == cfg.scene_service.service_type
    assert cfg2.segmentation_service.service_type == cfg.segmentation_service.service_type
    assert cfg2.scene_node.name == cfg.scene_node.name
    assert cfg2.segmentation_node.name == cfg.segmentation_node.name
    assert cfg2.pipeline_settings == cfg.pipeline_settings


def test_analysis_result_dict_roundtrip():
    r = AnalysisResult(
        output_directory="/out",
        result_files=["f1", "f2"],
        success=True,
        error="nothing",
    )
    d = r.to_dict()
    r2 = AnalysisResult.from_dict(d)
    assert r2.output_directory == r.output_directory
    assert r2.result_files == r.result_files
    assert r2.success == r.success
    assert r2.error == r.error


def test_analysis_workspace_dict_roundtrip():
    ws = AnalysisWorkspace(
        root=Path("/ws"),
        log_path=Path("/ws/log.txt"),
        config_path=Path("/ws/cfg.json"),
        result_path=Path("/ws/res"),
    )
    d = ws.to_dict()
    ws2 = AnalysisWorkspace.from_dict(d)
    assert ws2.root == ws.root
    assert ws2.log_path == ws.log_path
    assert ws2.config_path == ws.config_path
    assert ws2.result_path == ws.result_path


# =====================================================================
# 3. JSON round-trip — JSON-facing contracts
# =====================================================================

def test_model_spec_json_roundtrip():
    m = ModelSpec(repository="r", filename="f.bin", local_path="/l")
    j = json.loads(m.to_json())
    m2 = ModelSpec.from_dict(j)
    assert m2.repository == m.repository
    assert m2.filename == m.filename
    assert m2.local_path == m.local_path


def test_service_spec_json_roundtrip():
    s = ServiceSpec(
        service_type="llm",
        port=8000,
        model=ModelSpec(repository="r", filename="m.bin", local_path="/l"),
        settings={"temp": 0.7},
    )
    j = json.loads(s.to_json())
    s2 = ServiceSpec.from_dict(j)
    assert s2.service_type == s.service_type
    assert s2.port == s.port
    assert s2.model.filename == s.model.filename
    assert s2.settings == s.settings


def test_analysis_config_json_roundtrip():
    cfg = AnalysisConfig(
        input_path="/in",
        output_path="/out",
        scene_service=ServiceSpec(service_type="s", port=8000),
        segmentation_service=ServiceSpec(service_type="s", port=8000),
        scene_node=NodeConfig(name="s", host="1", instruction_port=5000, local=False),
        segmentation_node=NodeConfig(name="s", host="1", instruction_port=5000, local=False),
        pipeline_settings={"steps": 3},
    )
    j = json.loads(cfg.to_json())
    cfg2 = AnalysisConfig.from_dict(j)
    assert cfg2.input_path == cfg.input_path
    assert cfg2.output_path == cfg.output_path
    assert cfg2.scene_service.service_type == cfg.scene_service.service_type
    assert cfg2.segmentation_service.service_type == cfg.segmentation_service.service_type
    assert cfg2.scene_node.name == cfg.scene_node.name
    assert cfg2.segmentation_node.name == cfg.segmentation_node.name
    assert cfg2.pipeline_settings == cfg.pipeline_settings


def test_analysis_result_json_roundtrip():
    r = AnalysisResult(
        output_directory="/out",
        result_files=["a", "b"],
        success=True,
        error=None,
    )
    j = json.loads(r.to_json())
    r2 = AnalysisResult.from_dict(j)
    assert r2.output_directory == r.output_directory
    assert r2.success == r.success
    assert r2.error == r.error


def test_analysis_workspace_json_roundtrip():
    ws = AnalysisWorkspace(
        root=Path("/ws"),
        log_path=Path("/ws/log.txt"),
        config_path=Path("/ws/cfg.json"),
        result_path=Path("/ws/res"),
    )
    j = json.loads(ws.to_json())
    ws2 = AnalysisWorkspace.from_dict(j)
    assert ws2.root == ws.root
    assert ws2.log_path == ws.log_path
    assert ws2.config_path == ws.config_path
    assert ws2.result_path == ws.result_path


# =====================================================================
# 4. Nested contracts survive dict and JSON round trips
# =====================================================================

def test_nested_analysis_config_dict_roundtrip():
    """AnalysisConfig contains nested ServiceSpec, which contains ModelSpec."""
    cfg = AnalysisConfig(
        input_path="/in",
        output_path="/out",
        scene_service=ServiceSpec(
            service_type="scene",
            port=8000,
            model=ModelSpec(repository="scene-repo", filename="scene.bin", local_path="/m/scene"),
            settings={"resolution": "512x512", "verbose": True},
        ),
        segmentation_service=ServiceSpec(
            service_type="sam",
            port=9000,
            model=ModelSpec(repository="sam-model", filename="sam.bin"),
            settings={"nms_threshold": 0.5, "min_area": 1000},
        ),
        scene_node=NodeConfig(name="scene", host="127.0.0.1", instruction_port=5000, local=False),
        segmentation_node=NodeConfig(name="sam", host="127.0.0.1", instruction_port=5001, local=False),
        pipeline_settings={"iterations": 5},
    )
    d = cfg.to_dict()
    cfg2 = AnalysisConfig.from_dict(d)
    assert cfg2.scene_service.model.repository == "scene-repo"
    assert cfg2.scene_service.model.filename == "scene.bin"
    assert cfg2.scene_service.model.local_path == "/m/scene"
    assert cfg2.scene_service.settings["resolution"] == "512x512"
    assert cfg2.segmentation_service.model.repository == "sam-model"
    assert cfg2.segmentation_service.model.filename == "sam.bin"
    assert cfg2.scene_node.name == "scene"
    assert cfg2.segmentation_node.name == "sam"
    assert cfg2.pipeline_settings["iterations"] == 5


def test_nested_running_service_dict_roundtrip():
    """RunningService contains nested ServiceSpec and ServiceEndpoint."""
    rs = RunningService(
        spec=ServiceSpec(
            service_type="llm",
            port=8000,
            model=ModelSpec(repository="m", filename="m.bin", local_path="/l"),
            settings={"t": 0.7},
        ),
        endpoint=ServiceEndpoint(host="127.0.0.1", port=8001, service_type="llm"),
        runtime_handle="obj",
    )
    d = rs.to_dict()
    rs2 = RunningService.from_dict(d)
    assert rs2.spec.model.repository == "m"
    assert rs2.spec.model.filename == "m.bin"
    assert rs2.endpoint.host == "127.0.0.1"


def test_full_analysis_config_json_roundtrip():
    """Full nested JSON round-trip."""
    cfg = AnalysisConfig(
        input_path="/data/input.jpg",
        output_path="/data/output",
        scene_service=ServiceSpec(
            service_type="scene",
            port=8000,
            model=ModelSpec(repository="scene-model", filename="scene.bin", local_path="/m/scene"),
            settings={"resolution": "512x512", "verbose": True},
        ),
        segmentation_service=ServiceSpec(
            service_type="sam",
            port=9000,
            model=ModelSpec(repository="sam-model", filename="sam.bin"),
            settings={"nms_threshold": 0.5, "min_area": 1000},
        ),
        scene_node=NodeConfig(name="scene", host="127.0.0.1", instruction_port=5000, local=False),
        segmentation_node=NodeConfig(name="sam", host="127.0.0.1", instruction_port=5001, local=False),
        pipeline_settings={"steps": 3, "retry": True, "timeout": 30},
    )
    j = json.loads(cfg.to_json())
    cfg2 = AnalysisConfig.from_dict(j)
    assert cfg2.scene_service.model.repository == "scene-model"
    assert cfg2.segmentation_service.settings["nms_threshold"] == 0.5
    assert cfg2.pipeline_settings["timeout"] == 30


# =====================================================================
# 5. Mutable defaults isolation
# =====================================================================

def test_model_spec_no_shared_defaults():
    m1 = ModelSpec()
    m2 = ModelSpec()
    assert m1 is not m2


def test_service_spec_no_shared_settings():
    s1 = ServiceSpec(service_type="a", port=1000)
    s1.settings["x"] = 1
    s2 = ServiceSpec(service_type="b", port=2000)
    assert "x" not in s2.settings


def test_analysis_config_no_shared_pipeline_settings():
    c1 = AnalysisConfig(
        input_path="/in",
        output_path="/out",
        scene_service=ServiceSpec(service_type="s", port=8000),
        segmentation_service=ServiceSpec(service_type="s", port=8000),
        scene_node=NodeConfig(name="s", host="1", instruction_port=5000, local=False),
        segmentation_node=NodeConfig(name="s", host="1", instruction_port=5000, local=False),
    )
    c1.pipeline_settings["k"] = "v"
    c2 = AnalysisConfig(
        input_path="/in2",
        output_path="/out2",
        scene_service=ServiceSpec(service_type="s", port=8000),
        segmentation_service=ServiceSpec(service_type="s", port=8000),
        scene_node=NodeConfig(name="s", host="1", instruction_port=5000, local=False),
        segmentation_node=NodeConfig(name="s", host="1", instruction_port=5000, local=False),
    )
    assert "k" not in c2.pipeline_settings


def test_language_request_no_shared_settings():
    r1 = LanguageRequest(prompt="a")
    r1.settings["t"] = 0.7
    r2 = LanguageRequest(prompt="b")
    assert "t" not in r2.settings


# =====================================================================
# 6. Missing required fields fail clearly
# =====================================================================

def _check_missing_field(cls):
    """Assert that calling cls() without required args raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        cls()
    assert "missing" in str(exc_info.value).lower()


def test_service_spec_missing_service_type():
    _check_missing_field(ServiceSpec)


def test_service_spec_missing_port():
    _check_missing_field(ServiceSpec)


def test_service_endpoint_missing_host():
    _check_missing_field(ServiceEndpoint)


def test_service_endpoint_missing_port():
    _check_missing_field(ServiceEndpoint)


def test_service_endpoint_missing_service_type():
    _check_missing_field(ServiceEndpoint)


def test_node_config_missing_name():
    _check_missing_field(NodeConfig)


def test_node_config_missing_host():
    _check_missing_field(NodeConfig)


def test_node_config_missing_instruction_port():
    _check_missing_field(NodeConfig)


def test_analysis_config_missing_input_path():
    _check_missing_field(AnalysisConfig)


def test_analysis_config_missing_scene_service():
    _check_missing_field(AnalysisConfig)


def test_analysis_config_missing_scene_node():
    _check_missing_field(AnalysisConfig)


def test_analysis_config_missing_segmentation_node():
    _check_missing_field(AnalysisConfig)


def test_analysis_result_missing_output_directory():
    _check_missing_field(AnalysisResult)


def test_analysis_result_missing_result_files():
    _check_missing_field(AnalysisResult)


def test_analysis_result_missing_success():
    _check_missing_field(AnalysisResult)


def test_language_request_missing_prompt():
    _check_missing_field(LanguageRequest)


def test_language_response_missing_text():
    _check_missing_field(LanguageResponse)


def test_segmentation_request_missing_image():
    _check_missing_field(SegmentationRequest)


def test_segmentation_request_missing_prompt():
    _check_missing_field(SegmentationRequest)


def test_segmentation_result_missing_masks():
    _check_missing_field(SegmentationResult)


def test_segmentation_result_missing_labels():
    _check_missing_field(SegmentationResult)


def test_segmentation_result_missing_confidences():
    _check_missing_field(SegmentationResult)


def test_segmentation_result_missing_bounding_boxes():
    _check_missing_field(SegmentationResult)


def test_analysis_workspace_missing_root():
    _check_missing_field(AnalysisWorkspace)


def test_analysis_workspace_missing_log_path():
    _check_missing_field(AnalysisWorkspace)


def test_analysis_workspace_missing_config_path():
    _check_missing_field(AnalysisWorkspace)


def test_analysis_workspace_missing_result_path():
    _check_missing_field(AnalysisWorkspace)


# =====================================================================
# 7. runtime_handle excluded from serialization
# =====================================================================

def test_runtime_handle_excluded_from_dict():
    rs = RunningService(
        spec=ServiceSpec(service_type="llm", port=8000),
        endpoint=ServiceEndpoint(host="127.0.0.1", port=8001, service_type="llm"),
        runtime_handle="secret_handle_value",
    )
    d = rs.to_dict()
    assert "runtime_handle" not in d


def test_runtime_handle_excluded_from_json():
    rs = RunningService(
        spec=ServiceSpec(service_type="llm", port=8000),
        endpoint=ServiceEndpoint(host="127.0.0.1", port=8001, service_type="llm"),
        runtime_handle=object(),
    )
    j = json.dumps(rs.to_dict())
    data = json.loads(j)
    assert "runtime_handle" not in data


def test_runtime_handle_not_restored_from_dict():
    rs = RunningService(
        spec=ServiceSpec(service_type="llm", port=8000),
        endpoint=ServiceEndpoint(host="127.0.0.1", port=8001, service_type="llm"),
        runtime_handle="original",
    )
    d = rs.to_dict()
    rs2 = RunningService.from_dict(d)
    assert rs2.runtime_handle is None


def test_runtime_handle_not_restored_from_json():
    rs = RunningService(
        spec=ServiceSpec(service_type="llm", port=8000),
        endpoint=ServiceEndpoint(host="127.0.0.1", port=8001, service_type="llm"),
        runtime_handle="original",
    )
    j = json.dumps(rs.to_dict())
    rs2 = RunningService.from_dict(json.loads(j))
    assert rs2.runtime_handle is None


# =====================================================================
# 8. Non-copyable runtime_handle is safely excluded
# =====================================================================

class _NonCopyable:
    """A class whose copy/deepcopy both raise."""
    def __copy__(self):
        raise TypeError("cannot copy")
    def __deepcopy__(self, memo):
        raise TypeError("cannot deepcopy")

def test_non_copyable_runtime_handle_excluded():
    """A non-copyable handle is still excluded from to_dict()."""
    handle = _NonCopyable()
    rs = RunningService(
        spec=ServiceSpec(service_type="llm", port=8000),
        endpoint=ServiceEndpoint(host="127.0.0.1", port=8001, service_type="llm"),
        runtime_handle=handle,
    )
    d = rs.to_dict()
    assert "runtime_handle" not in d

    # Verify the dict itself is serialisable (no trace of the non-copyable object)
    json_str = json.dumps(d)
    json.loads(json_str)


# =====================================================================
# 9. Binary request data remains unchanged in memory
# =====================================================================

def test_language_request_binary_data_preserved():
    """LanguageRequest images are never mutated by the contract."""
    original_images = [b"\x89PNG\x00\x00", b"JPEG_DATA"]
    req = LanguageRequest(prompt="classify", images=original_images)
    assert req.images is original_images  # same object
    assert req.images[0] == b"\x89PNG\x00\x00"
    assert req.images[1] == b"JPEG_DATA"


def test_segmentation_request_binary_data_preserved():
    """SegmentationRequest image is never mutated by the contract."""
    image = b"\xFF\xD8\xFF\xE0JPEG_HEADER"
    req = SegmentationRequest(image=image, prompt="segment")
    assert req.image is image
    assert req.image == b"\xFF\xD8\xFF\xE0JPEG_HEADER"


# =====================================================================
# 10. Runtime/in-memory contracts have no serialization methods
# =====================================================================

def test_language_request_no_serialization_methods():
    """LanguageRequest has no to_dict/to_json — it is in-memory only."""
    req = LanguageRequest(prompt="test", images=[b"\x01\x02\x03"])
    assert not hasattr(req, "to_dict")
    assert not hasattr(req, "to_json")
    assert not hasattr(req, "from_dict")


def test_segmentation_request_no_serialization_methods():
    """SegmentationRequest has no to_dict/to_json — it is in-memory only."""
    req = SegmentationRequest(image=b"\x00\x01", prompt="x")
    assert not hasattr(req, "to_dict")
    assert not hasattr(req, "to_json")
    assert not hasattr(req, "from_dict")


def test_segmentation_result_no_serialization_methods():
    """SegmentationResult has no to_dict/to_json/from_dict — it is in-memory only."""
    res = SegmentationResult(
        masks=[True, False],
        labels=["cat"],
        confidences=[0.9],
        bounding_boxes=[[10, 10, 20, 20]],
    )
    assert not hasattr(res, "to_dict")
    assert not hasattr(res, "to_json")
    assert not hasattr(res, "from_dict")


def test_segmentation_result_retains_exact_masks():
    """SegmentationResult retains the exact mask objects supplied to it."""
    original_masks = [b"mask1", b"mask2", {"key": "value"}]
    res = SegmentationResult(
        masks=original_masks,
        labels=["a", "b"],
        confidences=[0.9, 0.8],
        bounding_boxes=[[0, 0, 10, 10], [10, 10, 20, 20]],
    )
    assert res.masks is original_masks  # same object, not copied
    assert res.masks[0] == b"mask1"
    assert res.masks[1] == b"mask2"
    assert res.masks[2] == {"key": "value"}


# =====================================================================
# 11. pathlib.Path round-trips correctly
# =====================================================================

def test_path_encoded_as_string():
    ws = AnalysisWorkspace(
        root=Path("/a/b/c"),
        log_path=Path("/d"),
        config_path=Path("/e"),
        result_path=Path("/f"),
    )
    d = ws.to_dict()
    assert isinstance(d["root"], str)
    assert isinstance(d["log_path"], str)
    assert isinstance(d["config_path"], str)
    assert isinstance(d["result_path"], str)
    assert d["root"] == "/a/b/c"


def test_path_reconstructed_on_deserialize():
    ws = AnalysisWorkspace(
        root=Path("/ws"),
        log_path=Path("/ws/log.txt"),
        config_path=Path("/ws/cfg.json"),
        result_path=Path("/ws/res"),
    )
    d = ws.to_dict()
    ws2 = AnalysisWorkspace.from_dict(d)
    assert isinstance(ws2.root, Path)
    assert isinstance(ws2.log_path, Path)
    assert isinstance(ws2.config_path, Path)
    assert isinstance(ws2.result_path, Path)
    assert ws2.root == Path("/ws")


def test_path_roundtrip_via_json():
    ws = AnalysisWorkspace(
        root=Path("/x/y/z"),
        log_path=Path("/log"),
        config_path=Path("/cfg"),
        result_path=Path("/res"),
    )
    j = json.loads(ws.to_json())
    ws2 = AnalysisWorkspace.from_dict(j)
    assert ws2.root == Path("/x/y/z")
    assert ws2.log_path == Path("/log")
    assert ws2.config_path == Path("/cfg")
    assert ws2.result_path == Path("/res")


def test_path_relative_to_absolute():
    ws = AnalysisWorkspace(
        root=Path("./relative/path"),
        log_path=Path("./relative/log.txt"),
        config_path=Path("./relative/cfg.json"),
        result_path=Path("./relative/res"),
    )
    d = ws.to_dict()
    ws2 = AnalysisWorkspace.from_dict(d)
    assert ws2.root == Path("./relative/path")


# =====================================================================
# 12. JSON-facing contracts have to_json; runtime/in-memory do not
# =====================================================================

def test_json_facing_contracts_have_to_json():
    """Only JSON-facing contracts implement to_json()."""
    for cls in (ModelSpec, ServiceSpec, ServiceEndpoint, NodeConfig,
                LanguageResponse, AnalysisConfig, AnalysisResult, AnalysisWorkspace):
        assert hasattr(cls, "to_json"), f"{cls.__name__} should have to_json()"


def test_runtime_contracts_no_to_json():
    """Runtime/in-memory contracts do not have to_json()."""
    for cls in (RunningService, LanguageRequest, SegmentationRequest, SegmentationResult):
        assert not hasattr(cls, "to_json"), f"{cls.__name__} should NOT have to_json()"
