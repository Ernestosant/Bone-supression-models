# Provenance

The original public project assets and historical checkpoints are not treated as sufficient
reproducibility evidence for this repository.

## Dataset

- Source: Kaggle `hmchuong/xray-bone-shadow-supression`.
- License listed by Kaggle: CC0/Public Domain.
- Primary paired inputs: `JSRT/JSRT`.
- Primary paired targets: `BSE_JSRT/BSE_JSRT`.
- Excluded from primary evaluation: Kaggle `augmented/`.

## Checkpoints

The historical U-Net checkpoint URL returned 404 during repository review, so retrained-v1 artifacts
must be treated as new model artifacts, not as recovered original weights.

Each retrained checkpoint must have:

- SHA256 checksum.
- Training manifest.
- Dataset split file.
- GPU evaluation metrics for the primary results, with CPU inference support retained for users.
- Public Google Drive download link.

## Notebooks

The original training notebooks are not present in this repository. The reproducible entrypoint is
now `notebooks/kaggle_retrain_bone_suppression.ipynb`, which records the commands needed to rebuild
the retrained-v1 artifacts from the public Kaggle dataset.

Historical local notebooks reviewed on 2026-05-11 (`Unet_MSO.ipynb` and `pix2pix.ipynb`) showed
that the original experiments first converted raw JSRT/BSE files through OpenCV 8-bit loading,
`255 - image`, and `cv2.equalizeHist`. Earlier retrained-v1 artifacts created without that exact
path are superseded and remain only as provenance for the failed visual QA pass.
