# Model Checkpoints

Model weights are intentionally not committed to this repository. Place downloaded checkpoints in
`models/checkpoints/` or another local directory and pass the path to the CLI or Gradio app.

The canonical model metadata lives in `configs/model_registry.json`.

| Key | Framework | Expected file | Current status |
| --- | --- | --- | --- |
| `gan_mso2` | TensorFlow/Keras | `gan_mso2_retrained_v1.keras` | Trained/evaluated; Drive upload pending |
| `unet_resnet50` | FastAI | `unet_resnet50_retrained_v1.pkl` | Trained/evaluated; Drive upload pending |

Large checkpoint files should remain outside Git history. After a Kaggle retraining run, upload
checkpoints to Google Drive, publish direct download links, and verify SHA256 values against the
registry before setting `available` to `true`.
