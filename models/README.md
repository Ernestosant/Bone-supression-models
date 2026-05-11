# Model Checkpoints

Model weights are intentionally not committed to this repository. Place downloaded checkpoints in
`models/checkpoints/` or another local directory and pass the path to the CLI or Gradio app.

The canonical model metadata lives in `configs/model_registry.json`.

| Key | Framework | Expected file | Current status |
| --- | --- | --- | --- |
| `gan_mso2` | TensorFlow/Keras | `gan_mso2.h5` | Google Drive link available |
| `unet_resnet50` | FastAI | `unet_resnet50.pkl` | Original Google Drive link returns 404 |

Large checkpoint files should remain outside Git history.
