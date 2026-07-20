from __future__ import annotations

import builtins

import numpy as np
import pytest

from core.services.sam_runtime import _load_predictor, _UltralyticsPredictor, resolve_sam_device


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeMasks:
    def __init__(self):
        self.data = FakeTensor([[[0, 1], [1, 1]]])


class FakeResult:
    masks = FakeMasks()
    boxes = None
    names = {}


class FakePredictor:
    model = object()

    def __init__(self):
        self.image = None
        self.text = None
        self.reset = False

    def set_image(self, image):
        self.image = image

    def __call__(self, *, text):
        self.text = text
        return [FakeResult()]

    def reset_image(self):
        self.reset = True


def test_semantic_predictor_uses_set_image_then_text_prompt():
    predictor = FakePredictor()
    wrapper = _UltralyticsPredictor(predictor)
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    payload = wrapper.predict(image, "car", 0.25)

    assert predictor.image is image
    assert predictor.text == ["car"]
    assert predictor.reset is True
    assert payload[0]["label"] == "car"
    assert payload[0]["confidence"] == 0.25
    assert payload[0]["box"] is None
    assert payload[0]["mask"].tolist() == [[0, 1], [1, 1]]


class SetupPredictor(FakePredictor):
    model = None

    def __init__(self):
        super().__init__()
        self.setup_count = 0
        self.model = None

    def setup_model(self, verbose=False):
        self.setup_count += 1
        self.model = object()


class FailingPredictor(FakePredictor):
    def __call__(self, *, text):
        raise RuntimeError("inference exploded")


def test_predictor_is_initialized_once_and_stays_warm():
    predictor = SetupPredictor()
    wrapper = _UltralyticsPredictor(predictor)
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    wrapper.predict(image, "car", 0.25)
    wrapper.predict(image, "person", 0.25)

    assert predictor.setup_count == 1


def test_predictor_resets_image_after_inference_failure():
    predictor = FailingPredictor()
    wrapper = _UltralyticsPredictor(predictor)

    with pytest.raises(RuntimeError, match="inference exploded"):
        wrapper.predict(np.zeros((2, 2, 3), dtype=np.uint8), "car", 0.25)

    assert predictor.reset is True


class FakeTorch:
    def __init__(self, *, cuda: bool, mps: bool):
        self.cuda = type("Cuda", (), {"is_available": staticmethod(lambda: cuda)})()
        self.backends = type("Backends", (), {"mps": type("Mps", (), {"is_available": staticmethod(lambda: mps)})()})()


def test_automatic_device_selection_prefers_cuda_then_mps_then_cpu():
    assert resolve_sam_device("auto", torch_module=FakeTorch(cuda=True, mps=True)) == "cuda"
    assert resolve_sam_device("auto", torch_module=FakeTorch(cuda=False, mps=True)) == "mps"
    assert resolve_sam_device("auto", torch_module=FakeTorch(cuda=False, mps=False)) == "cpu"


def test_explicit_unavailable_device_fails_clearly():
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        resolve_sam_device("cuda", torch_module=FakeTorch(cuda=False, mps=False))
    with pytest.raises(ValueError, match="MPS is unavailable"):
        resolve_sam_device("mps", torch_module=FakeTorch(cuda=False, mps=False))


def test_production_predictor_requires_existing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _load_predictor(tmp_path / "missing.pt")


def test_production_predictor_requires_ultralytics(monkeypatch, tmp_path):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"checkpoint")
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "ultralytics.models.sam":
            raise ImportError("missing ultralytics")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(RuntimeError, match="Ultralytics"):
        _load_predictor(checkpoint)
