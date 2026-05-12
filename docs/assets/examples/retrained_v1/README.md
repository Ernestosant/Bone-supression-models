# Retrained v1 Examples

This directory contains generated comparison panels from the deterministic test split.

The `*_gan_mso2_step1_input_output_target.png` panels are the README first-look examples. They
show input, corrected GAN MSO2 step-1 output, and paired BSE target after notebook-compatible MSO
preprocessing.

The `*_notebook_preprocessing_input_target.png` panels are preprocessing sanity checks created after
reviewing the historical MSO notebooks. They show the input X-ray and paired BSE target after
OpenCV 8-bit loading, `255 - image`, grayscale histogram equalization, and RGB expansion.

The step panels are produced by:

```bash
bone-suppression-repro step-comparison-examples --count 3 --steps 0,1,2,3,4,5
```

Each step panel contains input, target, and autoregressive outputs for steps 0-5. Current step
panels were regenerated from the corrected MSO retraining runs. Do not place the historical static
image here.
