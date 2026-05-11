# Bone Suppression Models Documentation

This documentation describes the research context, model registry, inference workflow, and
development process for the Bone Suppression Models repository.

## Intended Audience

- Researchers evaluating deep learning approaches for chest X-ray bone suppression.
- Engineers packaging experimental models into reproducible inference tools.
- Students documenting medical imaging experiments for professional review.

## Project Goals

- Keep inference code reusable instead of locking all behavior inside a notebook or demo script.
- Make model availability explicit through a registry.
- Separate dataset, model, inference, and development documentation.
- Avoid committing large datasets or checkpoints to Git.

## Current Capabilities

- Gradio demo for single-image inference.
- CLI entrypoint for scriptable inference.
- Shared preprocessing utilities for RGB conversion, histogram equalization, resizing, and
  normalization.
- Unit and smoke tests that do not require real checkpoint files.

## Documentation Map

- [Dataset](dataset.md): source dataset, access, and preprocessing assumptions.
- [Models](models.md): architecture summaries, checkpoint status, and limitations.
- [Inference](inference.md): CLI and Gradio usage.
- [Development](development.md): setup, tests, code style, and contribution workflow.
