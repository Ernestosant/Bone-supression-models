# Model Checkpoints

Model weights are intentionally not committed to this repository. Place downloaded checkpoints in
`models/checkpoints/` or another local directory and pass the path to the CLI or Gradio app.

The canonical model metadata lives in `configs/model_registry.json`.

| Key | Framework | Expected file | Current status |
| --- | --- | --- | --- |
| `gan_mso2` | TensorFlow/Keras | `gan_mso2_retrained_v1.keras` | Trained/evaluated; checkpoint not redistributed |
| `unet_resnet50` | FastAI | `unet_resnet50_retrained_v1.pkl` | Trained/evaluated; checkpoint not redistributed |

Large checkpoint files should remain outside Git history. Checkpoints are not redistributed here due
to size/storage constraints; metrics, hashes, manifests, scripts, visual panels, and reproducibility
instructions are provided. If a public artifact release is added later, publish direct download
links and verify SHA256 values against the registry before setting `available` to `true`.
