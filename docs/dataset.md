# Dataset

The original project references the Kaggle dataset
[Chest Xray Bone Shadow Suppression](https://www.kaggle.com/datasets/hmchuong/xray-bone-shadow-supression).
The dataset contains paired frontal chest X-ray images and corresponding bone-suppressed targets
generated with dual-energy subtraction.

## Access

1. Create or sign in to a Kaggle account.
2. Review the dataset license and terms on Kaggle.
3. Download the dataset locally outside this repository, for example under `data/raw/`.
4. Keep raw and processed data out of Git history.

## Dataset Notes

- Image pairs are reported as 1024 x 1024 pixels in the original README.
- The source dataset includes a larger augmented set and a smaller non-augmented set.
- The retrained-v1 pipeline uses paired non-augmented `JSRT/JSRT` and `BSE_JSRT/BSE_JSRT` images.
- The Kaggle `augmented/` directory is excluded from primary metrics to avoid leakage.

## Deterministic Split

Create the canonical split with:

```bash
bone-suppression-repro prepare-splits \
  --dataset-root data/raw/xray-bone-shadow-supression \
  --output training_runs/retrained_v1/splits.json \
  --seed 2026
```

The default split is 70% train, 15% validation, and 15% test after sorting paired filenames and
shuffling with seed `2026`.

## Preprocessing Assumptions

The corrected retraining path follows the historical MSO notebooks (`Unet_MSO.ipynb` and
`pix2pix.ipynb`) before training, evaluation, and example generation:

- Read each PNG with OpenCV default flags, which yields an 8-bit 3-channel image.
- Apply the intensity complement `255 - image`.
- Convert to grayscale, apply `cv2.equalizeHist`, then expand back to RGB.
- Resize/normalize model inputs after this notebook-compatible preprocessing step.
- Use a temporary TIFF handoff for FastAI U-Net inference because `test_dl` expects file-based input.

This matters because `BSE_JSRT` target PNG files are stored as 16-bit images. Reading them through
Pillow/FastAI directly changes the numeric range relative to the original notebooks and produced
misleading visual examples in the first retrained-v1 pass.

## Research-Use Considerations

Bone-suppressed images should be treated as model-generated derivatives, not replacements for
source radiographs. Any publication or presentation should report the dataset split, checkpoint,
preprocessing, and evaluation protocol used to generate examples or metrics.
