"""Gradio interface for interactive bone suppression inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bone_suppression.inference import run_inference
from bone_suppression.registry import load_model_registry


def predict_from_ui(
    image: np.ndarray,
    model_key: str,
    checkpoint_path: str,
    steps: int | float | None = None,
) -> np.ndarray:
    """Validate UI inputs and run inference."""
    if image is None:
        raise ValueError("Upload a chest X-ray image before running inference.")
    if not model_key:
        raise ValueError("Select a model before running inference.")
    if not checkpoint_path or not str(checkpoint_path).strip():
        raise ValueError("Provide a local checkpoint path before running inference.")

    normalized_steps = int(steps) if steps is not None else None
    return run_inference(model_key, Path(checkpoint_path), image, steps=normalized_steps)


def create_demo():
    """Build the Gradio Blocks application."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional env.
        raise RuntimeError("Install Gradio to run the demo application.") from exc

    registry = load_model_registry()
    choices = [(spec.display_name, spec.key) for spec in registry.values()]

    with gr.Blocks(title="Bone Suppression Models") as demo:
        gr.Markdown("# Bone Suppression Models")
        gr.Markdown(
            "Upload a chest X-ray, select a local checkpoint, and generate a "
            "bone-suppressed output image. Model weights are not stored in this repository."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="Input chest X-ray", type="numpy")
                model_key = gr.Dropdown(
                    choices=choices,
                    value="gan_mso2",
                    label="Model",
                )
                checkpoint_path = gr.Textbox(
                    label="Local checkpoint path",
                    placeholder="models/checkpoints/gan_mso2.h5",
                )
                steps = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=2,
                    step=1,
                    label="Inference steps",
                )
                run_button = gr.Button("Run inference", variant="primary")
            output = gr.Image(label="Bone-suppressed output", type="numpy")

        run_button.click(
            fn=predict_from_ui,
            inputs=[image, model_key, checkpoint_path, steps],
            outputs=output,
        )

    return demo


def main() -> None:
    create_demo().launch()
