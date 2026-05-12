"""Training routines for reproducible Kaggle retraining."""

from __future__ import annotations

import random
import tempfile
import time
from dataclasses import asdict, dataclass
from os import getenv
from pathlib import Path
from typing import Any

import numpy as np

from bone_suppression.artifacts import build_manifest, write_json
from bone_suppression.dataset import DATASET_SLUG, DEFAULT_SEED, ImagePair, load_splits
from bone_suppression.preprocessing import (
    LEGACY_MSO_PREPROCESSING,
    read_legacy_mso_image,
    resize_to_square,
    save_legacy_mso_image,
)


@dataclass(frozen=True)
class TrainConfig:
    model_key: str
    dataset_root: Path
    splits_path: Path
    output_dir: Path
    epochs: int = 50
    batch_size: int = 4
    image_size: int = 256
    learning_rate: float = 2e-4
    seed: int = DEFAULT_SEED
    limit: int | None = None
    pretrained: bool = True


def train_model(config: TrainConfig) -> dict[str, Any]:
    """Train one supported model and return its manifest payload."""
    if config.model_key == "gan_mso2":
        return train_gan_mso2(config)
    if config.model_key == "unet_resnet50":
        return train_unet_resnet50(config)
    raise ValueError(f"Unsupported training model: {config.model_key}")


def train_gan_mso2(config: TrainConfig) -> dict[str, Any]:
    """Train a compact Pix2Pix-style GAN and export a Keras generator."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Install TensorFlow requirements before training gan_mso2.") from exc

    _set_reproducible_seeds(config.seed)
    tf.random.set_seed(config.seed)

    splits = load_splits(config.splits_path, config.dataset_root)
    train_pairs = _limit_pairs(splits["train"], config.limit)
    valid_pairs = _limit_pairs(splits["validation"], config.limit)
    train_ds = _tf_pair_dataset(
        train_pairs,
        config.image_size,
        config.batch_size,
        shuffle=True,
        seed=config.seed,
    )
    valid_ds = _tf_pair_dataset(
        valid_pairs,
        config.image_size,
        config.batch_size,
        shuffle=False,
        seed=config.seed,
    )

    generator = _build_tf_generator(config.image_size)
    discriminator = _build_tf_discriminator(config.image_size)
    gen_optimizer = tf.keras.optimizers.Adam(config.learning_rate, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.Adam(config.learning_rate, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    lambda_l1 = 100.0

    @tf.function
    def train_step(source, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated = generator(source, training=True)
            real_logits = discriminator([source, target], training=True)
            fake_logits = discriminator([source, generated], training=True)
            gen_gan_loss = bce(tf.ones_like(fake_logits), fake_logits)
            gen_l1_loss = tf.reduce_mean(tf.abs(target - generated))
            gen_loss = gen_gan_loss + lambda_l1 * gen_l1_loss
            disc_real_loss = bce(tf.ones_like(real_logits), real_logits)
            disc_fake_loss = bce(tf.zeros_like(fake_logits), fake_logits)
            disc_loss = disc_real_loss + disc_fake_loss

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        return gen_loss, disc_loss

    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        gen_losses = []
        disc_losses = []
        for source, target in train_ds:
            gen_loss, disc_loss = train_step(source, target)
            gen_losses.append(float(gen_loss.numpy()))
            disc_losses.append(float(disc_loss.numpy()))
        val_l1 = _tf_validation_l1(generator, valid_ds, tf)
        epoch_seconds = time.perf_counter() - epoch_start
        record = {
            "epoch": float(epoch + 1),
            "generator_loss": float(np.mean(gen_losses)),
            "discriminator_loss": float(np.mean(disc_losses)),
            "validation_l1": val_l1,
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)
        print(
            "GAN epoch "
            f"{epoch + 1}/{config.epochs}: "
            f"gen_loss={record['generator_loss']:.4f} "
            f"disc_loss={record['discriminator_loss']:.4f} "
            f"val_l1={record['validation_l1']:.4f} "
            f"seconds={epoch_seconds:.2f}",
            flush=True,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "gan_mso2_retrained_v1.keras"
    generator.save(checkpoint_path)
    write_json(config.output_dir / "gan_mso2_history.json", {"history": history})
    hyperparameters = _config_payload(config) | {
        "lambda_l1": lambda_l1,
        "preprocessing": LEGACY_MSO_PREPROCESSING,
        "historical_reference": "pix2pix.ipynb preprocessing",
    }
    manifest = build_manifest(
        model_key=config.model_key,
        checkpoint_path=checkpoint_path,
        dataset_slug=DATASET_SLUG,
        split_path=config.splits_path,
        seed=config.seed,
        hyperparameters=hyperparameters,
    )
    write_json(config.output_dir / "gan_mso2_manifest.json", manifest)
    return manifest


def train_unet_resnet50(config: TrainConfig) -> dict[str, Any]:
    """Train a FastAI U-Net with a ResNet50 encoder and export a learner."""
    try:
        import torch
        from fastai.vision.all import (
            DataBlock,
            ImageBlock,
            IndexSplitter,
            MSELossFlat,
            Normalize,
            NormType,
            Resize,
            ResizeMethod,
            aug_transforms,
            imagenet_stats,
            resnet50,
            unet_learner,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("Install FastAI requirements before training unet_resnet50.") from exc

    _set_reproducible_seeds(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    splits = load_splits(config.splits_path, config.dataset_root)
    train_pairs = _limit_pairs(splits["train"], config.limit)
    valid_pairs = _limit_pairs(splits["validation"], config.limit)
    cache_root = _legacy_cache_root(config)
    train_pairs = _prepare_legacy_mso_cache(train_pairs, cache_root, config.image_size)
    valid_pairs = _prepare_legacy_mso_cache(valid_pairs, cache_root, config.image_size)
    all_pairs = train_pairs + valid_pairs
    valid_indices = list(range(len(train_pairs), len(all_pairs)))

    block = DataBlock(
        blocks=(ImageBlock, ImageBlock),
        get_x=pair_input_path,
        get_y=pair_target_path,
        splitter=IndexSplitter(valid_indices),
        item_tfms=Resize(config.image_size, ResizeMethod.Crop),
        batch_tfms=[
            *aug_transforms(max_zoom=1.0, flip_vert=True, do_flip=True, max_rotate=10),
            Normalize.from_stats(*imagenet_stats),
        ],
    )
    dls = block.dataloaders(all_pairs, bs=config.batch_size)
    dls.c = 3
    weight_decay = 1e-3
    y_range = (-3.0, 3.0)
    self_attention = _env_flag("BONE_SUPPRESSION_UNET_SELF_ATTENTION", default=False)
    learner = unet_learner(
        dls,
        resnet50,
        n_out=3,
        pretrained=config.pretrained,
        norm_type=NormType.Weight,
        self_attention=self_attention,
        y_range=y_range,
        loss_func=MSELossFlat(),
    )
    learner.fit_one_cycle(
        config.epochs,
        lr_max=config.learning_rate,
        pct_start=0.8,
        wd=weight_decay,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = config.output_dir / "unet_resnet50_retrained_v1.pkl"
    learner.export(checkpoint_path)
    manifest = build_manifest(
        model_key=config.model_key,
        checkpoint_path=checkpoint_path,
        dataset_slug=DATASET_SLUG,
        split_path=config.splits_path,
        seed=config.seed,
        hyperparameters=_config_payload(config)
        | {
            "preprocessing": LEGACY_MSO_PREPROCESSING,
            "weight_decay": weight_decay,
            "y_range": list(y_range),
            "self_attention": self_attention,
            "fastai_resize_method": "crop",
            "fastai_normalization": "imagenet_stats",
            "historical_reference": "Unet_MSO.ipynb preprocessing",
            "p100_note": (
                "self_attention defaults to false because FastAI spectral norm self-attention "
                "failed on Kaggle Tesla P100 with CUBLAS_STATUS_NOT_SUPPORTED"
            ),
        },
    )
    write_json(config.output_dir / "unet_resnet50_manifest.json", manifest)
    return manifest


def _limit_pairs(pairs: list[ImagePair], limit: int | None) -> list[ImagePair]:
    return pairs[:limit] if limit is not None else pairs


def pair_input_path(pair: ImagePair) -> Path:
    """Top-level getter so FastAI exports remain importable."""
    return pair.input_path


def pair_target_path(pair: ImagePair) -> Path:
    """Top-level getter so FastAI exports remain importable."""
    return pair.target_path


def _config_payload(config: TrainConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["dataset_root"] = str(config.dataset_root)
    payload["splits_path"] = str(config.splits_path)
    payload["output_dir"] = str(config.output_dir)
    return payload


def _set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _load_pair_arrays(pair: ImagePair, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    source = _load_normalized_image(pair.input_path, image_size)
    target = _load_normalized_image(pair.target_path, image_size)
    return source, target


def _load_normalized_image(path: Path, image_size: int) -> np.ndarray:
    image = resize_to_square(read_legacy_mso_image(path), image_size)
    array = image.astype(np.float32)
    return array / 127.5 - 1.0


def _legacy_cache_root(config: TrainConfig) -> Path:
    configured = getenv("BONE_SUPPRESSION_PREPROCESSED_CACHE_DIR")
    if configured:
        return Path(configured)
    return (
        Path(tempfile.gettempdir())
        / "bone_suppression_legacy_mso"
        / config.model_key
        / f"seed_{config.seed}_size_{config.image_size}"
    )


def _prepare_legacy_mso_cache(
    pairs: list[ImagePair],
    cache_root: Path,
    image_size: int,
) -> list[ImagePair]:
    cached_pairs: list[ImagePair] = []
    for pair in pairs:
        input_path = cache_root / "source" / f"{pair.id}.png"
        target_path = cache_root / "target" / f"{pair.id}.png"
        if not input_path.exists():
            save_legacy_mso_image(pair.input_path, input_path, image_size=image_size)
        if not target_path.exists():
            save_legacy_mso_image(pair.target_path, target_path, image_size=image_size)
        cached_pairs.append(ImagePair(pair.id, input_path, target_path))
    return cached_pairs


def _env_flag(name: str, default: bool) -> bool:
    value = getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tf_pair_dataset(
    pairs: list[ImagePair],
    image_size: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    import tensorflow as tf

    def generator():
        for pair in pairs:
            yield _load_pair_arrays(pair, image_size)

    output_signature = (
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(image_size, image_size, 3), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(pairs),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _tf_validation_l1(generator, dataset, tf) -> float:
    losses = []
    for source, target in dataset:
        generated = generator(source, training=False)
        losses.append(float(tf.reduce_mean(tf.abs(target - generated)).numpy()))
    return float(np.mean(losses)) if losses else float("nan")


def _build_tf_generator(image_size: int):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    down1 = _conv_block(inputs, 64, strides=2)
    down2 = _conv_block(down1, 128, strides=2)
    down3 = _conv_block(down2, 256, strides=2)
    bottleneck = _conv_block(down3, 512, strides=2)
    up1 = _up_block(bottleneck, down3, 256)
    up2 = _up_block(up1, down2, 128)
    up3 = _up_block(up2, down1, 64)
    x = tf.keras.layers.UpSampling2D()(up3)
    outputs = tf.keras.layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
    return tf.keras.Model(inputs, outputs, name="gan_mso2_generator")


def _build_tf_discriminator(image_size: int):
    import tensorflow as tf

    source = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    target = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Concatenate()([source, target])
    x = _conv_block(x, 64, strides=2, batch_norm=False)
    x = _conv_block(x, 128, strides=2)
    x = _conv_block(x, 256, strides=2)
    outputs = tf.keras.layers.Conv2D(1, 4, padding="same")(x)
    return tf.keras.Model([source, target], outputs, name="gan_mso2_discriminator")


def _conv_block(inputs, filters: int, strides: int = 1, batch_norm: bool = True):
    import tensorflow as tf

    x = tf.keras.layers.Conv2D(
        filters,
        4,
        strides=strides,
        padding="same",
        use_bias=not batch_norm,
    )(inputs)
    if batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.LeakyReLU(0.2)(x)


def _up_block(inputs, skip, filters: int):
    import tensorflow as tf

    x = tf.keras.layers.UpSampling2D()(inputs)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    return tf.keras.layers.Concatenate()([x, skip])
