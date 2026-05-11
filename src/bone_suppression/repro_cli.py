"""Reproducibility CLI for dataset splits, training, metrics, and examples."""

from __future__ import annotations

import argparse
from pathlib import Path

from bone_suppression.dataset import (
    DEFAULT_SEED,
    DEFAULT_SOURCE_SUBDIR,
    DEFAULT_TARGET_SUBDIR,
    DEFAULT_TRAIN_FRACTION,
    DEFAULT_VALIDATION_FRACTION,
    write_splits,
)
from bone_suppression.evaluation import (
    build_comparison_examples,
    build_step_comparison_examples,
    evaluate_checkpoint,
    evaluate_checkpoint_steps,
    merge_metric_files,
)
from bone_suppression.training import TrainConfig, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible bone-suppression workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-splits", help="Create deterministic dataset splits.")
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    prepare.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    prepare.add_argument("--source-subdir", default=DEFAULT_SOURCE_SUBDIR)
    prepare.add_argument("--target-subdir", default=DEFAULT_TARGET_SUBDIR)

    train = subparsers.add_parser("train", help="Train one supported model.")
    train.add_argument("--model", required=True, choices=["gan_mso2", "unet_resnet50"])
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--splits", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--image-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--limit", type=int, default=None)
    train.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train U-Net without pretrained ResNet50 weights.",
    )

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one checkpoint on a split.")
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--dataset-root", required=True)
    evaluate.add_argument("--splits", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--split", default="test")
    evaluate.add_argument("--device", choices=["auto", "cpu"], default="cpu")
    evaluate.add_argument("--steps", type=int, default=None)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--no-save-predictions", action="store_true")

    evaluate_steps = subparsers.add_parser(
        "evaluate-steps",
        help="Evaluate autoregressive steps for one checkpoint.",
    )
    evaluate_steps.add_argument("--model", required=True)
    evaluate_steps.add_argument("--checkpoint", required=True)
    evaluate_steps.add_argument("--dataset-root", required=True)
    evaluate_steps.add_argument("--splits", required=True)
    evaluate_steps.add_argument("--output-dir", required=True)
    evaluate_steps.add_argument("--split", default="test")
    evaluate_steps.add_argument("--device", choices=["auto", "cpu"], default="cpu")
    evaluate_steps.add_argument("--steps", default="0,1,2,3,4,5")
    evaluate_steps.add_argument("--limit", type=int, default=None)
    evaluate_steps.add_argument("--no-save-predictions", action="store_true")

    merge = subparsers.add_parser("merge-metrics", help="Merge per-model metrics into one file.")
    merge.add_argument("--output", required=True)
    merge.add_argument("metrics", nargs="+")

    examples = subparsers.add_parser(
        "comparison-examples",
        help="Create input/target/GAN/U-Net comparison panels.",
    )
    examples.add_argument("--dataset-root", required=True)
    examples.add_argument("--splits", required=True)
    examples.add_argument("--output-dir", required=True)
    examples.add_argument("--split", default="test")
    examples.add_argument("--count", type=int, default=3)
    examples.add_argument("--gan-predictions", required=True)
    examples.add_argument("--unet-predictions", required=True)

    step_examples = subparsers.add_parser(
        "step-comparison-examples",
        help="Create input/target/step comparison panels for one model.",
    )
    step_examples.add_argument("--dataset-root", required=True)
    step_examples.add_argument("--splits", required=True)
    step_examples.add_argument("--prediction-root", required=True)
    step_examples.add_argument("--output-dir", required=True)
    step_examples.add_argument("--model-label", required=True)
    step_examples.add_argument("--steps", default="0,1,2,3,4,5")
    step_examples.add_argument("--split", default="test")
    step_examples.add_argument("--count", type=int, default=3)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-splits":
        write_splits(
            output_path=args.output,
            dataset_root=args.dataset_root,
            seed=args.seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            source_subdir=args.source_subdir,
            target_subdir=args.target_subdir,
        )
        return 0

    if args.command == "train":
        train_model(
            TrainConfig(
                model_key=args.model,
                dataset_root=Path(args.dataset_root),
                splits_path=Path(args.splits),
                output_dir=Path(args.output_dir),
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                limit=args.limit,
                pretrained=not args.no_pretrained,
            )
        )
        return 0

    if args.command == "evaluate":
        evaluate_checkpoint(
            model_key=args.model,
            checkpoint_path=args.checkpoint,
            dataset_root=args.dataset_root,
            splits_path=args.splits,
            output_dir=args.output_dir,
            split=args.split,
            device=args.device,
            steps=args.steps,
            limit=args.limit,
            save_predictions=not args.no_save_predictions,
        )
        return 0

    if args.command == "evaluate-steps":
        evaluate_checkpoint_steps(
            model_key=args.model,
            checkpoint_path=args.checkpoint,
            dataset_root=args.dataset_root,
            splits_path=args.splits,
            output_dir=args.output_dir,
            split=args.split,
            device=args.device,
            steps_values=_parse_steps(args.steps),
            limit=args.limit,
            save_predictions=not args.no_save_predictions,
        )
        return 0

    if args.command == "merge-metrics":
        merge_metric_files(args.metrics, args.output)
        return 0

    if args.command == "comparison-examples":
        build_comparison_examples(
            dataset_root=args.dataset_root,
            splits_path=args.splits,
            prediction_dirs={
                "GAN MSO2": args.gan_predictions,
                "U-Net ResNet50": args.unet_predictions,
            },
            output_dir=args.output_dir,
            split=args.split,
            count=args.count,
        )
        return 0

    if args.command == "step-comparison-examples":
        build_step_comparison_examples(
            dataset_root=args.dataset_root,
            splits_path=args.splits,
            prediction_root=args.prediction_root,
            output_dir=args.output_dir,
            model_label=args.model_label,
            steps_values=_parse_steps(args.steps),
            split=args.split,
            count=args.count,
        )
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


def _parse_steps(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
