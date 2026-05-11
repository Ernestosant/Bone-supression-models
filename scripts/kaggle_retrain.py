"""Kaggle entrypoint for retrained-v1 bone suppression artifacts."""

# ruff: noqa: E402,I001

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if not (REPO_ROOT / "src").exists():
    REPO_ROOT = REPO_ROOT.parent
if (REPO_ROOT / "src").exists():
    sys.path.insert(0, str(REPO_ROOT / "src"))

from bone_suppression.artifacts import sha256_file  # noqa: E402
from bone_suppression.dataset import (  # noqa: E402
    DATASET_SLUG,
    DEFAULT_SEED,
    DEFAULT_SOURCE_SUBDIR,
    DEFAULT_TARGET_SUBDIR,
    write_splits,
)
from bone_suppression.evaluation import (  # noqa: E402
    build_step_comparison_examples,
    evaluate_checkpoint_steps,
)
from bone_suppression.training import TrainConfig, train_model  # noqa: E402


DATASET_ROOT = Path(
    os.getenv("BONE_SUPPRESSION_DATASET_ROOT", "/kaggle/input/xray-bone-shadow-supression")
)
RUN_DIR = Path(os.getenv("BONE_SUPPRESSION_RUN_DIR", "/kaggle/working/training_runs/retrained_v1"))
EPOCHS = int(os.getenv("BONE_SUPPRESSION_EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("BONE_SUPPRESSION_BATCH_SIZE", "4"))
IMAGE_SIZE = int(os.getenv("BONE_SUPPRESSION_IMAGE_SIZE", "256"))
LEARNING_RATE = float(os.getenv("BONE_SUPPRESSION_LEARNING_RATE", "0.0002"))
LIMIT = os.getenv("BONE_SUPPRESSION_LIMIT")
LIMIT_VALUE = int(LIMIT) if LIMIT else None
EVAL_LIMIT = os.getenv("BONE_SUPPRESSION_EVAL_LIMIT")
EVAL_LIMIT_VALUE = int(EVAL_LIMIT) if EVAL_LIMIT else None
EVAL_DEVICE = os.getenv("BONE_SUPPRESSION_EVAL_DEVICE", "auto")
RUN_MODE = os.getenv("BONE_SUPPRESSION_RUN_MODE", "train_eval")
EVAL_STEPS = [
    int(step.strip())
    for step in os.getenv("BONE_SUPPRESSION_EVAL_STEPS", "0,1,2,3,4,5").split(",")
    if step.strip()
]
MODELS = [
    model.strip()
    for model in os.getenv("BONE_SUPPRESSION_MODELS", "gan_mso2,unet_resnet50").split(",")
    if model.strip()
]


def main() -> int:
    run_start = time.perf_counter()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "Starting retrained-v1 run "
        f"mode={RUN_MODE} models={MODELS} epochs={EPOCHS} "
        f"batch_size={BATCH_SIZE} image_size={IMAGE_SIZE} eval_steps={EVAL_STEPS}",
        flush=True,
    )
    dataset_root = resolve_dataset_root(DATASET_ROOT)
    write_environment_report(RUN_DIR / "environment.json", dataset_root)

    splits_path = RUN_DIR / "splits.json"
    write_splits(splits_path, dataset_root, seed=DEFAULT_SEED)

    checkpoint_paths = {
        "gan_mso2": Path(
            os.getenv(
                "BONE_SUPPRESSION_GAN_MSO2_CHECKPOINT",
                str(RUN_DIR / "gan_mso2" / "gan_mso2_retrained_v1.keras"),
            )
        ),
        "unet_resnet50": Path(
            os.getenv(
                "BONE_SUPPRESSION_UNET_RESNET50_CHECKPOINT",
                str(RUN_DIR / "unet_resnet50" / "unet_resnet50_retrained_v1.pkl"),
            )
        ),
    }
    metric_payloads = {}

    for model_key in MODELS:
        model_start = time.perf_counter()
        model_dir = RUN_DIR / model_key
        train_seconds = None
        if RUN_MODE != "eval_only":
            print(f"Training {model_key} started.", flush=True)
            train_model(
                TrainConfig(
                    model_key=model_key,
                    dataset_root=dataset_root,
                    splits_path=splits_path,
                    output_dir=model_dir,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    image_size=IMAGE_SIZE,
                    learning_rate=LEARNING_RATE,
                    limit=LIMIT_VALUE,
                )
            )
            train_seconds = time.perf_counter() - model_start
            print(f"Training {model_key} finished in {train_seconds:.2f} seconds.", flush=True)
        checkpoint_path = resolve_checkpoint_path(model_key, checkpoint_paths[model_key])
        if RUN_MODE == "train_only":
            metric_payloads[model_key] = {
                "checkpoint_filename": checkpoint_path.name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "train_seconds": train_seconds,
                "step_results": [],
            }
            continue

        evaluation_dir = RUN_DIR / "evaluation"
        print(
            f"Evaluating {model_key} autoregressive steps {EVAL_STEPS} on {EVAL_DEVICE}.",
            flush=True,
        )
        metrics = evaluate_checkpoint_steps(
            model_key=model_key,
            checkpoint_path=checkpoint_path,
            dataset_root=dataset_root,
            splits_path=splits_path,
            output_dir=evaluation_dir,
            device=EVAL_DEVICE,
            split="test",
            steps_values=EVAL_STEPS,
            limit=EVAL_LIMIT_VALUE,
        )
        metrics["train_seconds"] = train_seconds
        metric_payloads[model_key] = metrics

    metrics_path = RUN_DIR / "metrics.json"
    write_step_metrics_summary(metrics_path, metric_payloads)

    for model_key in MODELS:
        if not metric_payloads.get(model_key, {}).get("step_results"):
            continue
        model_label = "GAN MSO2" if model_key == "gan_mso2" else "U-Net ResNet50"
        build_step_comparison_examples(
            dataset_root=dataset_root,
            splits_path=splits_path,
            prediction_root=RUN_DIR / "evaluation" / "predictions" / model_key,
            output_dir=RUN_DIR / "examples",
            model_label=model_label,
            steps_values=EVAL_STEPS,
            split="test",
            count=3,
        )

    run_seconds = time.perf_counter() - run_start
    write_run_summary(RUN_DIR / "run_summary.json", metric_payloads, run_seconds)
    print("Retrained-v1 artifacts completed.")
    summary = summarize_step_payloads(metric_payloads)
    summary["run_seconds"] = run_seconds
    print(json.dumps(summary, indent=2))
    return 0


def resolve_dataset_root(preferred: Path) -> Path:
    """Resolve a Kaggle dataset mount or download the public dataset as fallback."""
    candidates = [preferred]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(path for path in input_root.iterdir() if path.is_dir())
        candidates.extend(path for path in input_root.glob("*/*") if path.is_dir())

    for candidate in candidates:
        if _looks_like_dataset_root(candidate):
            print(f"Using dataset root: {candidate}", flush=True)
            return candidate

    print(
        f"Dataset mount not found under {preferred} or /kaggle/input; trying kagglehub download.",
        flush=True,
    )
    try:
        import kagglehub
    except ImportError as exc:
        raise FileNotFoundError(
            "Dataset root was not mounted and kagglehub is unavailable for fallback download."
        ) from exc

    downloaded = Path(kagglehub.dataset_download(DATASET_SLUG))
    if _looks_like_dataset_root(downloaded):
        print(f"Using kagglehub dataset root: {downloaded}", flush=True)
        return downloaded
    for candidate in [downloaded, *downloaded.glob("*"), *downloaded.glob("*/*")]:
        if candidate.is_dir() and _looks_like_dataset_root(candidate):
            print(f"Using nested kagglehub dataset root: {candidate}", flush=True)
            return candidate
    raise FileNotFoundError(
        f"Could not resolve dataset root after kagglehub download: {downloaded}"
    )


def _looks_like_dataset_root(path: Path) -> bool:
    return (path / DEFAULT_SOURCE_SUBDIR).exists() and (path / DEFAULT_TARGET_SUBDIR).exists()


def resolve_checkpoint_path(model_key: str, preferred: Path) -> Path:
    if preferred.exists():
        print(f"Using checkpoint for {model_key}: {preferred}", flush=True)
        return preferred

    expected_names = {
        "gan_mso2": "gan_mso2_retrained_v1.keras",
        "unet_resnet50": "unet_resnet50_retrained_v1.pkl",
    }
    expected = expected_names[model_key]
    for root in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not root.exists():
            continue
        matches = sorted(root.rglob(expected))
        if matches:
            print(f"Using discovered checkpoint for {model_key}: {matches[0]}", flush=True)
            return matches[0]
    raise FileNotFoundError(f"Checkpoint not found for {model_key}: {preferred}")


def write_environment_report(path: Path, dataset_root: Path) -> None:
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "requested_dataset_root": str(DATASET_ROOT),
        "dataset_root": str(dataset_root),
        "run_dir": str(RUN_DIR),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "learning_rate": LEARNING_RATE,
        "limit": LIMIT_VALUE,
        "eval_limit": EVAL_LIMIT_VALUE,
        "models": MODELS,
        "run_mode": RUN_MODE,
        "eval_steps": EVAL_STEPS,
        "eval_device": EVAL_DEVICE,
        "nvidia_smi": run_optional(["nvidia-smi"]),
        "pip_freeze": run_optional([sys.executable, "-m", "pip", "freeze"]),
        "dataset_root_exists": dataset_root.exists(),
    }
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def run_optional(command: list[str]) -> str:
    executable = shutil.which(command[0])
    if executable is None and command[0] != sys.executable:
        return "unavailable"
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    except Exception as exc:  # pragma: no cover - diagnostic only.
        return f"failed: {exc}"
    return (result.stdout + result.stderr).strip()


def write_run_summary(path: Path, metric_payloads: dict, run_seconds: float) -> None:
    payload = {
        "models": MODELS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "limit": LIMIT_VALUE,
        "eval_limit": EVAL_LIMIT_VALUE,
        "eval_device": EVAL_DEVICE,
        "run_seconds": run_seconds,
        "metrics": {
            model_key: {
                "steps": {
                    str(result["steps"]): result["aggregate"]
                    for result in metrics["step_results"]
                },
                "train_seconds": metrics.get("train_seconds"),
            }
            for model_key, metrics in metric_payloads.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_step_metrics_summary(path: Path, metric_payloads: dict) -> None:
    payload = {
        "schema_version": 1,
        "models": [
            {
                "model_key": model_key,
                "checkpoint_filename": metrics.get("checkpoint_filename"),
                "checkpoint_sha256": metrics.get("checkpoint_sha256"),
                "split": metrics.get("split"),
                "device": metrics.get("device"),
                "step_results": [
                    {
                        "steps": result["steps"],
                        "counts": result["counts"],
                        "aggregate": result["aggregate"],
                    }
                    for result in metrics["step_results"]
                ],
            }
            for model_key, metrics in metric_payloads.items()
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def summarize_step_payloads(metric_payloads: dict) -> dict:
    return {
        model_key: {
            str(result["steps"]): result["aggregate"]
            for result in metrics["step_results"]
        }
        for model_key, metrics in metric_payloads.items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
