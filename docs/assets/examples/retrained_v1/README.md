# Retrained v1 Examples

This directory contains generated comparison panels from the deterministic test split.

The `*_notebook_preprocessing_input_target.png` panels are preprocessing sanity checks created
after reviewing the historical MSO notebooks. They show the input X-ray and paired BSE target after
OpenCV 8-bit loading, `255 - image`, grayscale histogram equalization, and RGB expansion.

The step panels are produced by:

```bash
bone-suppression-repro step-comparison-examples --count 3 --steps 0,1,2,3,4,5
```

Each step panel contains input, target, and autoregressive outputs for steps 0-5. Step panels from
the first retrained-v1 pass are superseded until the models are retrained with notebook-compatible
preprocessing. Do not place the historical static image here.
