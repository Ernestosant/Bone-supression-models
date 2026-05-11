from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from bone_suppression import evaluation
from bone_suppression.dataset import write_splits


def _write_png(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def _make_dataset(root: Path, count: int = 8) -> None:
    for index in range(count):
        filename = f"JPCLN{index:03d}.png"
        _write_png(root / "JSRT" / "JSRT" / filename, value=index)
        _write_png(root / "BSE_JSRT" / "BSE_JSRT" / filename, value=index)


def test_evaluate_checkpoint_writes_metrics_and_predictions(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    splits_path = tmp_path / "splits.json"
    write_splits(splits_path, dataset_root)
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")

    calls = {"load": 0, "predict": 0}

    def fake_load_model(spec, checkpoint_path, device="auto"):
        calls["load"] += 1
        return "fake-model"

    def fake_predict_model(model, spec, image, steps=None):
        calls["predict"] += 1
        return image

    monkeypatch.setattr(evaluation, "load_model", fake_load_model)
    monkeypatch.setattr(evaluation, "predict_model", fake_predict_model)

    payload = evaluation.evaluate_checkpoint(
        model_key="gan_mso2",
        checkpoint_path=checkpoint,
        dataset_root=dataset_root,
        splits_path=splits_path,
        output_dir=tmp_path / "evaluation",
        device="cpu",
        limit=1,
    )

    assert payload["counts"] == {"evaluated": 1}
    assert calls == {"load": 1, "predict": 1}
    assert (tmp_path / "evaluation" / "gan_mso2_test_metrics.json").exists()
    assert len(list((tmp_path / "evaluation" / "predictions" / "gan_mso2").glob("*.png"))) == 1


def test_evaluate_checkpoint_steps_handles_zero_and_model_steps(tmp_path, monkeypatch) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    splits_path = tmp_path / "splits.json"
    write_splits(splits_path, dataset_root)
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    calls = {"load": 0, "predict": []}

    def fake_load_model(spec, checkpoint_path, device="auto"):
        calls["load"] += 1
        return "fake-model"

    def fake_predict_model(model, spec, image, steps=None):
        calls["predict"].append(steps)
        return np.zeros((256, 256, 3), dtype=np.uint8)

    monkeypatch.setattr(evaluation, "load_model", fake_load_model)
    monkeypatch.setattr(evaluation, "predict_model", fake_predict_model)

    payload = evaluation.evaluate_checkpoint_steps(
        model_key="gan_mso2",
        checkpoint_path=checkpoint,
        dataset_root=dataset_root,
        splits_path=splits_path,
        output_dir=tmp_path / "evaluation",
        steps_values=(0, 1, 2),
        device="cpu",
        limit=1,
    )

    assert [result["steps"] for result in payload["step_results"]] == [0, 1, 2]
    assert calls["load"] == 1
    assert calls["predict"] == [1, 2]
    assert (tmp_path / "evaluation" / "gan_mso2_test_steps_metrics.json").exists()
