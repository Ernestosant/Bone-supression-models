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

The inference package applies a conservative preprocessing sequence based on the original demo:

- Convert grayscale, single-channel, or RGBA inputs to RGB.
- Apply histogram equalization to image intensity.
- Resize GAN inputs to 256 x 256 when the uploaded image is larger than the target size.
- Normalize GAN inputs to `[-1, 1]`.
- Use a temporary TIFF handoff for FastAI U-Net inference because `test_dl` expects file-based input.

## Research-Use Considerations

Bone-suppressed images should be treated as model-generated derivatives, not replacements for
source radiographs. Any publication or presentation should report the dataset split, checkpoint,
preprocessing, and evaluation protocol used to generate examples or metrics.
