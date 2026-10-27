"""Dataset preparation helpers for single-class character detection labs."""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
READ_ONLY_SHARED_ROOT = Path("/data/shared/project2026")
MANIFEST_VERSION = 1


@dataclass(frozen=True)
class DetectionPair:
    """A page image and its matching character-level annotation file."""

    image_path: Path
    label_path: Path
    relative_image: Path


def discover_detection_pairs(
    source: Path, *, image_dir: str = "images", label_dir: str = "label_char"
) -> list[DetectionPair]:
    """Find image/label pairs while preserving paths below the image root."""

    source = source.resolve()
    image_root = source / image_dir
    label_root = source / label_dir
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(
            f"Expected {image_dir}/ and {label_dir}/ directories under {source}"
        )

    pairs = []
    unmatched_images = []
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_image = image_path.relative_to(image_root)
        label_path = label_root / relative_image.with_suffix(".txt")
        if not label_path.is_file():
            unmatched_images.append(relative_image.as_posix())
            continue
        pairs.append(DetectionPair(image_path, label_path, relative_image))
    if unmatched_images:
        preview = ", ".join(unmatched_images[:5])
        raise ValueError(
            f"Found {len(unmatched_images)} images without matching {label_dir} labels: {preview}"
        )
    return pairs


def convert_xyxy_label(
    label_path: Path, *, image_width: int, image_height: int
) -> str:
    """Convert ``class x1 y1 x2 y2`` rows to single-class YOLO rows."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    converted = []
    for line_number, fields in _label_rows(label_path):
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        values = _finite_floats(fields, label_path, line_number)
        _, x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= image_width and 0 <= y1 < y2 <= image_height):
            raise ValueError(
                f"{label_path}:{line_number}: pixel box is outside the image or non-positive"
            )
        x = (x1 + x2) / (2 * image_width)
        y = (y1 + y2) / (2 * image_height)
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height
        converted.append(f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
    return "\n".join(converted) + ("\n" if converted else "")


def normalize_yolo_label(label_path: Path) -> str:
    """Validate YOLO rows and force every valid object class to zero."""

    converted = []
    for line_number, fields in _label_rows(label_path):
        if len(fields) < 5:
            raise ValueError(f"{label_path}:{line_number}: expected at least 5 fields")
        values = _finite_floats(fields[:5], label_path, line_number)
        _, x, y, width, height = values
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(
                f"{label_path}:{line_number}: normalized coordinates must be in [0, 1] "
                "with positive width and height"
            )
        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
            raise ValueError(f"{label_path}:{line_number}: box exceeds image width")
        if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            raise ValueError(f"{label_path}:{line_number}: box exceeds image height")
        converted.append(f"0 {x:.6f} {y:.6f} {width:.6f} {height:.6f}")
    return "\n".join(converted) + ("\n" if converted else "")


def split_pairs(
    pairs: Iterable[DetectionPair], *, seed: int = 2026
) -> dict[str, list[DetectionPair]]:
    """Return a deterministic, exhaustive 80/10/10 split."""

    shuffled = sorted(pairs, key=lambda pair: pair.relative_image.as_posix())
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    if count < 10:
        raise ValueError("At least 10 matched pages are required for an 80/10/10 split")
    train_end = int(count * 0.8)
    val_end = train_end + int(count * 0.1)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_mth1000(
    source: Path,
    destination: Path,
    *,
    mode: Literal["practice", "full"],
    sample_size: int = 24,
    seed: int = 2026,
    materialize: Literal["symlink", "copy"] = "symlink",
) -> dict[str, object]:
    """Build a single-class YOLO view of MTH1000."""

    source = source.resolve()
    destination = destination.resolve()
    _validate_new_destination(source, destination)
    pairs = discover_detection_pairs(source)
    if mode == "practice":
        if len(pairs) < sample_size:
            raise RuntimeError(f"Found {len(pairs)} pairs; requested {sample_size}")
        selected = list(pairs)
        random.Random(seed).shuffle(selected)
        selected = sorted(selected[:sample_size], key=lambda p: p.relative_image.as_posix())
        split_map = {"all": selected}
        yaml_splits = {"train": "all", "val": "all", "test": "all"}
    elif mode == "full":
        split_map = split_pairs(pairs, seed=seed)
        yaml_splits = {name: name for name in ("train", "val", "test")}
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    entries = []
    destination.mkdir(parents=True)
    for split, split_pairs_ in split_map.items():
        for pair in split_pairs_:
            image_out = destination / "images" / split / pair.relative_image
            label_out = destination / "labels" / split / pair.relative_image.with_suffix(".txt")
            _materialize_image(pair.image_path, image_out, materialize)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(pair.image_path) as image:
                label_text = convert_xyxy_label(
                    pair.label_path, image_width=image.width, image_height=image.height
                )
            label_out.write_text(label_text, encoding="utf-8")
            entries.append(
                {
                    "split": split,
                    "image": pair.relative_image.as_posix(),
                    "source_label": pair.label_path.relative_to(source).as_posix(),
                    "boxes": len(label_text.splitlines()),
                }
            )

    manifest = {
        "version": MANIFEST_VERSION,
        "kind": "mth1000-single-class",
        "mode": mode,
        "source": str(source),
        "seed": seed,
        "size": len(entries),
        "splits": {name: len(value) for name, value in split_map.items()},
        "entries": entries,
    }
    _write_json(destination / "manifest.json", manifest)
    _write_data_yaml(destination, yaml_splits)
    return manifest


def prepare_yolo_view(
    source: Path,
    destination: Path,
    *,
    label_dir: str,
    split_files: dict[str, Path] | None = None,
    materialize: Literal["symlink", "copy"] = "symlink",
) -> dict[str, object]:
    """Create a standard images/labels view from existing YOLO character labels."""

    source = source.resolve()
    destination = destination.resolve()
    _validate_new_destination(source, destination)
    pairs = discover_detection_pairs(source, label_dir=label_dir)
    pair_by_relative = {pair.relative_image.as_posix(): pair for pair in pairs}
    pair_by_name: dict[str, list[DetectionPair]] = {}
    for pair in pairs:
        pair_by_name.setdefault(pair.relative_image.name, []).append(pair)

    if split_files:
        split_map = {
            split: _pairs_from_split_file(path, pair_by_relative, pair_by_name)
            for split, path in split_files.items()
        }
    else:
        split_map = {"test": pairs}
    _assert_disjoint(split_map)

    entries = []
    destination.mkdir(parents=True)
    for split, split_pairs_ in split_map.items():
        for pair in split_pairs_:
            image_out = destination / "images" / split / pair.relative_image
            label_out = destination / "labels" / split / pair.relative_image.with_suffix(".txt")
            _materialize_image(pair.image_path, image_out, materialize)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            label_text = normalize_yolo_label(pair.label_path)
            label_out.write_text(label_text, encoding="utf-8")
            entries.append(
                {
                    "split": split,
                    "image": pair.relative_image.as_posix(),
                    "source_label": pair.label_path.relative_to(source).as_posix(),
                    "boxes": len(label_text.splitlines()),
                }
            )

    manifest = {
        "version": MANIFEST_VERSION,
        "kind": "single-class-yolo-view",
        "source": str(source),
        "source_label_directory": label_dir,
        "size": len(entries),
        "splits": {name: len(value) for name, value in split_map.items()},
        "entries": entries,
    }
    _write_json(destination / "manifest.json", manifest)
    _write_data_yaml(destination, {name: name for name in split_map})
    return manifest


def _pairs_from_split_file(
    split_file: Path,
    pair_by_relative: dict[str, DetectionPair],
    pair_by_name: dict[str, list[DetectionPair]],
) -> list[DetectionPair]:
    selected = []
    for line_number, raw in enumerate(split_file.read_text(encoding="utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value)
        relative_candidates = [candidate.as_posix()]
        if "images" in candidate.parts:
            index = candidate.parts.index("images")
            relative_candidates.append(Path(*candidate.parts[index + 1 :]).as_posix())
        pair = next(
            (pair_by_relative[key] for key in relative_candidates if key in pair_by_relative),
            None,
        )
        if pair is None:
            matches = pair_by_name.get(candidate.name, [])
            if len(matches) == 1:
                pair = matches[0]
        if pair is None:
            raise ValueError(f"{split_file}:{line_number}: cannot uniquely match {value!r}")
        selected.append(pair)
    if len({pair.relative_image for pair in selected}) != len(selected):
        raise ValueError(f"{split_file}: contains duplicate images")
    return selected


def _assert_disjoint(split_map: dict[str, list[DetectionPair]]) -> None:
    ownership: dict[Path, str] = {}
    for split, pairs in split_map.items():
        for pair in pairs:
            previous = ownership.setdefault(pair.relative_image, split)
            if previous != split:
                raise ValueError(
                    f"{pair.relative_image} occurs in both {previous!r} and {split!r}"
                )


def _validate_new_destination(source: Path, destination: Path) -> None:
    shared = READ_ONLY_SHARED_ROOT.resolve()
    if destination == shared or shared in destination.parents:
        raise ValueError(f"Destination must not be inside read-only {shared}")
    if destination == source or source in destination.parents:
        raise ValueError("Destination must not be inside the source dataset")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty destination: {destination}")


def _materialize_image(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported materialization mode: {mode}")


def _write_data_yaml(destination: Path, splits: dict[str, str]) -> None:
    lines = [f"path: {destination}"]
    lines.extend(f"{name}: images/{directory}" for name, directory in splits.items())
    lines.extend(["nc: 1", "names:", "  0: character"])
    (destination / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _label_rows(path: Path) -> Iterable[tuple[int, list[str]]]:
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if fields:
            yield line_number, fields


def _finite_floats(fields: Iterable[str], path: Path, line_number: int) -> list[float]:
    try:
        values = [float(value) for value in fields]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: fields must be numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}:{line_number}: fields must be finite")
    return values


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
