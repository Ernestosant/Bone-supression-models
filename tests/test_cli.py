from __future__ import annotations

import numpy as np
from PIL import Image

from bone_suppression import cli


def test_cli_passes_device_to_inference(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(input_path)
    calls = {}

    def fake_run_inference(model_key, checkpoint_path, image, steps=None, device="auto"):
        calls["model_key"] = model_key
        calls["checkpoint_path"] = checkpoint_path
        calls["steps"] = steps
        calls["device"] = device
        return np.ones((4, 4, 3), dtype=np.uint8) * 255

    monkeypatch.setattr(cli, "run_inference", fake_run_inference)

    exit_code = cli.main(
        [
            "--model",
            "gan_mso2",
            "--checkpoint",
            "models/checkpoints/gan_mso2_retrained_v1.keras",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--steps",
            "1",
            "--device",
            "cpu",
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    assert calls["device"] == "cpu"
