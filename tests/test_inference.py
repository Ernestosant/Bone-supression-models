from __future__ import annotations

import numpy as np

from bone_suppression.inference import predict_model
from bone_suppression.registry import get_model_spec


class FakeGan:
    def __call__(self, model_input, training: bool = False):
        assert training is True
        batch, height, width, channels = model_input.shape
        assert batch == 1
        assert channels == 3
        return np.zeros((1, height, width, channels), dtype=np.float32)


class FakeDls:
    def test_dl(self, path: str):
        assert path.endswith(".tif")
        return ["fake-dl", path]


class FakeUnet:
    dls = FakeDls()

    def get_preds(self, dl):
        assert dl[0] == "fake-dl"
        return [np.zeros((3, 16, 16), dtype=np.float32)]


def test_predict_pix2pix_gan_smoke() -> None:
    image = np.zeros((300, 300, 3), dtype=np.uint8)

    prediction = predict_model(FakeGan(), get_model_spec("gan_mso2"), image, steps=1)

    assert prediction.shape == (256, 256, 3)
    assert prediction.dtype == np.uint8


def test_predict_unet_smoke() -> None:
    image = np.zeros((32, 32), dtype=np.uint8)

    prediction = predict_model(FakeUnet(), get_model_spec("unet_resnet50"), image, steps=1)

    assert prediction.shape == (16, 16, 3)
    assert prediction.dtype == np.uint8
