from __future__ import annotations

import numpy as np

from core.services.sam_runtime import _UltralyticsPredictor


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
