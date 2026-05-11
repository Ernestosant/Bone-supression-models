# Bone Suppression Models Documentation

This documentation describes the research context, reproducible training pipeline, model registry,
inference workflow, evaluation protocol, and development process.

## Intended Audience

- Researchers evaluating deep learning approaches for chest X-ray bone suppression.
- Engineers packaging experimental models into reproducible inference tools.
- Students documenting medical imaging experiments for professional review.

## Current Capabilities

- Deterministic JSRT/BSE_JSRT pairing and 70/15/15 split generation with seed `2026`.
- Kaggle retraining notebook for `gan_mso2` and `unet_resnet50`.
- Quantitative evaluation utilities for MAE, RMSE, PSNR, SSIM, and CPU inference time.
- Gradio and CLI inference with `auto` or CPU-only device selection.
- Registry fields for public checkpoint links, checksums, metrics, examples, and artifacts.
- Unit and smoke tests that do not require real checkpoint files.

## Documentation Map

- [Dataset](dataset.md): source dataset, access, split policy, and preprocessing assumptions.
- [Training](training.md): Kaggle and local retraining workflow.
- [Evaluation](evaluation.md): holdout metrics and example generation.
- [Results](results.md): final results table template and publication checklist.
- [Models](models.md): architecture summaries, checkpoint policy, and registry fields.
- [Inference](inference.md): CLI and Gradio usage.
- [Provenance](provenance.md): dataset, historical weights, and notebook provenance.
- [Development](development.md): setup, tests, code style, and contribution workflow.
