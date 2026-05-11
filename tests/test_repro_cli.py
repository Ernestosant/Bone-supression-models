from __future__ import annotations

from pathlib import Path

from bone_suppression import repro_cli


def test_train_command_builds_smoke_config(monkeypatch, tmp_path) -> None:
    calls = {}

    def fake_train_model(config):
        calls["config"] = config
        return {}

    monkeypatch.setattr(repro_cli, "train_model", fake_train_model)

    exit_code = repro_cli.main(
        [
            "train",
            "--model",
            "gan_mso2",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--splits",
            str(tmp_path / "splits.json"),
            "--output-dir",
            str(tmp_path / "run"),
            "--epochs",
            "1",
            "--limit",
            "4",
        ]
    )

    assert exit_code == 0
    assert calls["config"].epochs == 1
    assert calls["config"].limit == 4
    assert calls["config"].dataset_root == Path(tmp_path / "dataset")
