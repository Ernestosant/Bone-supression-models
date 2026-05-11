from __future__ import annotations

from pathlib import Path

from PIL import Image

from bone_suppression.dataset import (
    build_split_payload,
    discover_pairs,
    load_splits,
    write_splits,
)


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), "black").save(path)


def _make_dataset(root: Path, count: int = 10) -> None:
    for index in range(count):
        filename = f"JPCLN{index:03d}.png"
        _write_png(root / "JSRT" / "JSRT" / filename)
        _write_png(root / "BSE_JSRT" / "BSE_JSRT" / filename)
    _write_png(root / "augmented" / "augmented" / "source" / "0_0.png")


def test_discover_pairs_matches_non_augmented_jsrt_files(tmp_path) -> None:
    _make_dataset(tmp_path, count=3)

    pairs = discover_pairs(tmp_path)

    assert [pair.id for pair in pairs] == ["JPCLN000", "JPCLN001", "JPCLN002"]
    assert all("augmented" not in str(pair.input_path) for pair in pairs)


def test_split_payload_is_deterministic(tmp_path) -> None:
    _make_dataset(tmp_path, count=20)
    pairs = discover_pairs(tmp_path)

    first = build_split_payload(pairs, tmp_path, seed=2026)
    second = build_split_payload(pairs, tmp_path, seed=2026)

    assert first["splits"] == second["splits"]
    assert first["counts"] == {"total": 20, "train": 14, "validation": 3, "test": 3}


def test_write_and_load_splits_round_trip(tmp_path) -> None:
    _make_dataset(tmp_path, count=8)
    splits_path = tmp_path / "splits.json"

    write_splits(splits_path, tmp_path, seed=2026)
    splits = load_splits(splits_path, tmp_path)

    assert set(splits) == {"train", "validation", "test"}
    assert sum(len(items) for items in splits.values()) == 8
