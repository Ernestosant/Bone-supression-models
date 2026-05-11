from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from bone_suppression import training
from bone_suppression.dataset import ImagePair


def test_tf_pair_dataset_uses_configured_shuffle_seed(monkeypatch) -> None:
    class FakeDataset:
        def __init__(self):
            self.shuffle_seed = None
            self.batch_size = None
            self.prefetch_value = None

        def shuffle(self, buffer_size, seed, reshuffle_each_iteration):
            self.shuffle_seed = seed
            return self

        def batch(self, batch_size):
            self.batch_size = batch_size
            return self

        def prefetch(self, value):
            self.prefetch_value = value
            return self

    fake_dataset = FakeDataset()

    class FakeDatasetFactory:
        @staticmethod
        def from_generator(generator, output_signature):
            return fake_dataset

    fake_tf = SimpleNamespace(
        TensorSpec=lambda shape, dtype: (shape, dtype),
        float32="float32",
        data=SimpleNamespace(Dataset=FakeDatasetFactory, AUTOTUNE="autotune"),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    pair = ImagePair("sample", Path("input.png"), Path("target.png"))
    dataset = training._tf_pair_dataset(
        [pair],
        image_size=8,
        batch_size=2,
        shuffle=True,
        seed=1234,
    )

    assert dataset.shuffle_seed == 1234
    assert dataset.batch_size == 2
    assert dataset.prefetch_value == "autotune"
