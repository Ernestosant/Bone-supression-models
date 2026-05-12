# Model Checkpoints

Model weights are intentionally not committed to this repository. Place downloaded checkpoints in
`models/checkpoints/` or another local directory and pass the path to the CLI or Gradio app.

The canonical model metadata lives in `configs/model_registry.json`.

| Key | Framework | Expected file | Current status |
| --- | --- | --- | --- |
| `gan_mso2` | TensorFlow/Keras | `gan_mso2_retrained_v1.keras` | Corrected MSO retrain complete; public checkpoint available |
| `unet_resnet50` | FastAI | `unet_resnet50_retrained_v1.pkl` | Corrected MSO retrain complete; public checkpoint available |

Large checkpoint files should remain outside Git history. Corrected MSO checkpoints, metrics,
hashes, manifests, and visual panels are published in the
[`corrected-mso-v1`](https://github.com/Ernestosant/Bone-supression-models/releases/tag/corrected-mso-v1)
GitHub Release. The U-Net checkpoint is split into three parts; concatenate them in order and verify
the reconstructed `.pkl` against the registry SHA256.
