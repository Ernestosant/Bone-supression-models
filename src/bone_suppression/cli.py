"""Command line interface for single-image inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from bone_suppression.inference import run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bone suppression inference.")
    parser.add_argument(
        "--model",
        required=True,
        help="Model key from configs/model_registry.json.",
    )
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint file path.")
    parser.add_argument("--input", required=True, help="Input chest X-ray image path.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of iterative inference steps.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu"],
        default="auto",
        help="Inference device. Use 'cpu' for reproducible CPU-only loading.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image = np.asarray(Image.open(args.input).convert("RGB"))
    prediction = run_inference(
        args.model,
        args.checkpoint,
        image,
        steps=args.steps,
        device=args.device,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction).save(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
