"""Dataset preparation helpers for single-class character detection labs."""

from __future__ import annotations

import json
import math
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
READ_ONLY_SHARED_ROOT = Path("/data/shared/project2026")
CANONICAL_SHARED_MTH_DESTINATION = READ_ONLY_SHARED_ROOT / "mth-full-project-ws2026"
MANIFEST_VERSION = 1
PRACTICE_SPLITS = {"train": 18, "val": 2, "test": 4}
YOLO_BOUNDARY_TOLERANCE = 1e-3


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
    """Convert ``label x1 y1 x2 y2`` rows to single-class YOLO rows.

    The raw label may be a numeric class ID or a character glyph. It is
    ignored because this course dataset has one object class.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    converted = []
    for line_number, fields in _label_rows(label_path):
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        x1, y1, x2, y2 = _finite_floats(fields[1:], label_path, line_number)
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


def split_train_val(
    pairs: Iterable[DetectionPair], *, seed: int = 2026, val_fraction: float = 0.1
) -> dict[str, list[DetectionPair]]:
    """Return deterministic, exhaustive train/validation subsets."""

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    shuffled = sorted(pairs, key=lambda pair: pair.relative_image.as_posix())
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 2:
        raise ValueError("At least two pages are required for a train/validation split")
    val_count = max(1, round(len(shuffled) * val_fraction))
    train_count = len(shuffled) - val_count
    if train_count == 0:
        raise ValueError("The validation split leaves no training pages")
    return {"train": shuffled[:train_count], "val": shuffled[train_count:]}


def stage_mth_practice_sample(
    source: Path,
    destination: Path,
    *,
    seed: int = 2026,
) -> dict[str, object]:
    """Copy a fixed raw-label sample for the hands-on Week 3 exercise."""

    source = source.resolve()
    destination = destination.resolve()
    _validate_new_destination(source, destination)
    pairs = discover_detection_pairs(source)
    if len(pairs) < sum(PRACTICE_SPLITS.values()):
        raise RuntimeError("Practice staging requires at least 24 MTH1200 pairs")

    rng = random.Random(seed)
    selected = list(pairs)
    rng.shuffle(selected)
    selected = selected[: sum(PRACTICE_SPLITS.values())]

    destination.mkdir(parents=True)
    entries = []
    for pair in selected:
        image_out = destination / "images" / pair.relative_image
        label_relative = pair.relative_image.with_suffix(".txt")
        label_out = destination / "labels" / label_relative
        _materialize_file(pair.image_path, image_out, "copy")
        _materialize_file(pair.label_path, label_out, "copy")
        entries.append(
            {
                "image": pair.relative_image.as_posix(),
                "raw_label": label_relative.as_posix(),
                "source_image": pair.image_path.relative_to(source).as_posix(),
                "source_label": pair.label_path.relative_to(source).as_posix(),
            }
        )

    entries.sort(key=lambda entry: str(entry["image"]))
    manifest = {
        "version": MANIFEST_VERSION,
        "kind": "mth-practice-source-sample",
        "seed": seed,
        "source": str(source),
        "size": len(entries),
        "splits": dict(PRACTICE_SPLITS),
        "entries": entries,
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def prepare_mth_course_dataset(
    train_source: Path,
    test_source: Path,
    destination: Path,
    *,
    seed: int = 2026,
    materialize: Literal["symlink", "copy"] = "symlink",
    allow_shared_output: bool = False,
) -> dict[str, object]:
    """Prepare the canonical MTH1200/MTH1000 YOLO dataset view.

    Source ``labels/`` files must already be normalized, single-class YOLO labels.
    Zero-area source rows are omitted; all other rows are validated and copied beside
    their selected images.
    """

    train_source = train_source.resolve()
    test_source = test_source.resolve()
    destination = destination.resolve()
    _validate_new_destination(
        train_source, destination, allow_shared_output=allow_shared_output
    )
    _validate_new_destination(
        test_source, destination, allow_shared_output=allow_shared_output
    )
    train_val = split_train_val(
        discover_detection_pairs(train_source, label_dir="labels"), seed=seed
    )
    split_map = {
        **train_val,
        "test": discover_detection_pairs(test_source, label_dir="labels"),
    }
    sanitized_labels = {
        pair.label_path: _sanitize_single_class_yolo_label(pair.label_path)
        for pairs in split_map.values()
        for pair in pairs
    }

    destination.mkdir(parents=True)
    entries = []
    dropped_degenerate_boxes = 0
    for split, pairs in split_map.items():
        source = test_source if split == "test" else train_source
        collection = "mth1000" if split == "test" else "mth1200"
        for pair in pairs:
            image_out = destination / "images" / split / pair.relative_image
            label_out = destination / "labels" / split / pair.relative_image.with_suffix(".txt")
            _materialize_file(pair.image_path, image_out, materialize)
            label_text, source_boxes, dropped_boxes = sanitized_labels[pair.label_path]
            label_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.write_text(label_text, encoding="utf-8")
            dropped_degenerate_boxes += dropped_boxes
            entries.append(
                {
                    "split": split,
                    "collection": collection,
                    "image": pair.relative_image.as_posix(),
                    "source_label": pair.label_path.relative_to(source).as_posix(),
                    "source_boxes": source_boxes,
                    "boxes": source_boxes - dropped_boxes,
                    "dropped_degenerate_boxes": dropped_boxes,
                }
            )

    manifest = {
        "version": MANIFEST_VERSION,
        "kind": "mth-course-detection",
        "protocol": "MTH1200 90/10 train/val; full MTH1000 test",
        "seed": seed,
        "sources": {"mth1200": str(train_source), "mth1000": str(test_source)},
        "size": len(entries),
        "splits": {name: len(pairs) for name, pairs in split_map.items()},
        "dropped_degenerate_boxes": dropped_degenerate_boxes,
        "entries": entries,
    }
    _write_json(destination / "manifest.json", manifest)
    _write_data_yaml(destination, {name: name for name in ("train", "val", "test")})
    return manifest


def validate_mth_practice_dataset(
    dataset: Path, source_manifest: Path
) -> dict[str, object]:
    """Validate the student-built 18/2/4 practice dataset."""

    dataset = dataset.resolve()
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    expected = {Path(entry["image"]).as_posix() for entry in manifest["entries"]}
    problems: list[str] = []
    actual: set[str] = set()
    split_counts: dict[str, int] = {}
    for split, expected_count in PRACTICE_SPLITS.items():
        image_root = dataset / "images" / split
        label_root = dataset / "labels" / split
        images = [
            path for path in sorted(image_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ] if image_root.is_dir() else []
        split_counts[split] = len(images)
        if len(images) != expected_count:
            problems.append(f"{split}: expected {expected_count} images, found {len(images)}")
        for image_path in images:
            relative = image_path.relative_to(image_root)
            key = relative.as_posix()
            if key in actual:
                problems.append(f"page assigned more than once: {relative}")
            actual.add(key)
            label_path = label_root / relative.with_suffix(".txt")
            if not label_path.is_file():
                problems.append(f"missing label for {split}/{relative}")
                continue
            try:
                _validate_single_class_yolo_label(label_path)
            except ValueError as exc:
                problems.append(str(exc))
        orphan_labels = {
            path.relative_to(label_root).with_suffix("").as_posix()
            for path in label_root.rglob("*.txt")
        } - {
            path.relative_to(image_root).with_suffix("").as_posix() for path in images
        } if label_root.is_dir() else set()
        problems.extend(f"orphan label in {split}: {name}" for name in sorted(orphan_labels))

    if actual != expected:
        missing = expected - actual
        unexpected = actual - expected
        if missing:
            problems.append(f"missing staged pages: {sorted(missing)[:5]}")
        if unexpected:
            problems.append(f"pages not in the staged sample: {sorted(unexpected)[:5]}")
    try:
        _validate_detection_yaml(dataset / "data.yaml", dataset)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        problems.append(str(exc))
    if problems:
        raise ValueError("Practice dataset validation failed:\n- " + "\n- ".join(problems))
    return {"valid": True, "size": sum(split_counts.values()), "splits": split_counts}


def prepare_nomnaocr_archive(archive: Path, destination: Path) -> dict[str, object]:
    """Prepare every labeled Roboflow archive page as one external test set."""

    archive = archive.resolve()
    destination = destination.resolve()
    _validate_new_destination(archive, destination)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a ZIP archive: {archive}")

    entries = []
    seen_outputs: set[Path] = set()
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as bundle:
        names = set(bundle.namelist())
        for name in sorted(names):
            archive_path = Path(name)
            if archive_path.is_absolute() or ".." in archive_path.parts:
                raise ValueError(f"Unsafe archive member: {name}")
            if archive_path.suffix.lower() not in IMAGE_SUFFIXES or "images" not in archive_path.parts:
                continue
            image_index = archive_path.parts.index("images")
            relative = Path(*archive_path.parts[image_index + 1 :])
            prefix = archive_path.parts[:image_index]
            partition = prefix[-1] if prefix else "all"
            label_member = Path(*prefix, "labels", *relative.parts).with_suffix(".txt").as_posix()
            if label_member not in names:
                raise ValueError(f"Archive image has no matching label: {name}")
            output_relative = Path(partition) / relative
            if output_relative in seen_outputs:
                raise ValueError(f"Duplicate archive output path: {output_relative}")
            seen_outputs.add(output_relative)
            image_out = destination / "images" / "test" / output_relative
            label_out = destination / "labels" / "test" / output_relative.with_suffix(".txt")
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            image_out.write_bytes(bundle.read(name))
            label_out.write_bytes(bundle.read(label_member))
            label_text = normalize_yolo_label(label_out)
            label_out.write_text(label_text, encoding="utf-8")
            entries.append(
                {
                    "split": "test",
                    "archive_partition": partition,
                    "image": output_relative.as_posix(),
                    "source_image": name,
                    "source_label": label_member,
                    "boxes": len(label_text.splitlines()),
                }
            )
    if not entries:
        raise ValueError(f"No labeled images found in {archive}")
    manifest = {
        "version": MANIFEST_VERSION,
        "kind": "nomnaocr-external-test",
        "source_archive": str(archive),
        "size": len(entries),
        "splits": {"test": len(entries)},
        "entries": entries,
    }
    _write_json(destination / "manifest.json", manifest)
    _write_data_yaml(destination, {"test": "test"})
    return manifest


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


def _validate_new_destination(
    source: Path, destination: Path, *, allow_shared_output: bool = False
) -> None:
    shared = READ_ONLY_SHARED_ROOT.resolve()
    if destination == shared or shared in destination.parents:
        canonical = (shared / CANONICAL_SHARED_MTH_DESTINATION.name).resolve()
        if not allow_shared_output or destination != canonical:
            raise ValueError(
                "Destination must not be inside read-only "
                f"{shared}; only staff may prepare {canonical} with "
                "--allow-shared-output"
            )
    if destination == source or source in destination.parents:
        raise ValueError("Destination must not be inside the source dataset")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty destination: {destination}")


def _is_below_shared(path: Path) -> bool:
    shared = READ_ONLY_SHARED_ROOT.resolve()
    return path == shared or shared in path.parents


def _materialize_image(source: Path, destination: Path, mode: str) -> None:
    _materialize_file(source, destination, mode)


def _materialize_file(source: Path, destination: Path, mode: str) -> None:
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


def _validate_single_class_yolo_label(path: Path) -> None:
    for line_number, fields in _label_rows(path):
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected exactly 5 fields")
        values = _finite_floats(fields, path, line_number)
        class_id, x, y, width, height = values
        if class_id != 0:
            raise ValueError(f"{path}:{line_number}: class must be 0")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(
                f"{path}:{line_number}: normalized coordinates must be in [0, 1] "
                "with positive width and height"
            )
        if (
            x - width / 2 < -YOLO_BOUNDARY_TOLERANCE
            or x + width / 2 > 1 + YOLO_BOUNDARY_TOLERANCE
        ):
            raise ValueError(f"{path}:{line_number}: box exceeds image width")
        if (
            y - height / 2 < -YOLO_BOUNDARY_TOLERANCE
            or y + height / 2 > 1 + YOLO_BOUNDARY_TOLERANCE
        ):
            raise ValueError(f"{path}:{line_number}: box exceeds image height")


def _sanitize_single_class_yolo_label(path: Path) -> tuple[str, int, int]:
    """Return valid rows, total source rows, and omitted zero-area rows."""

    sanitized_rows = []
    source_boxes = 0
    dropped_boxes = 0
    for line_number, fields in _label_rows(path):
        source_boxes += 1
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected exactly 5 fields")
        values = _finite_floats(fields, path, line_number)
        class_id, x, y, width, height = values
        if class_id != 0:
            raise ValueError(f"{path}:{line_number}: class must be 0")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
            raise ValueError(
                f"{path}:{line_number}: normalized coordinates must be in [0, 1]"
            )
        if (
            x - width / 2 < -YOLO_BOUNDARY_TOLERANCE
            or x + width / 2 > 1 + YOLO_BOUNDARY_TOLERANCE
        ):
            raise ValueError(f"{path}:{line_number}: box exceeds image width")
        if (
            y - height / 2 < -YOLO_BOUNDARY_TOLERANCE
            or y + height / 2 > 1 + YOLO_BOUNDARY_TOLERANCE
        ):
            raise ValueError(f"{path}:{line_number}: box exceeds image height")
        if width == 0 or height == 0:
            dropped_boxes += 1
            continue
        sanitized_rows.append(" ".join(fields))
    label_text = "\n".join(sanitized_rows)
    if label_text:
        label_text += "\n"
    return label_text, source_boxes, dropped_boxes


def _validate_detection_yaml(path: Path, dataset: Path) -> None:
    if not path.is_file():
        raise ValueError(f"Missing data.yaml: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    expected = {name: f"images/{name}" for name in PRACTICE_SPLITS}
    for name, expected_path in expected.items():
        if value.get(name) != expected_path:
            raise ValueError(f"{path}: {name} must be {expected_path!r}")
    configured_path = Path(str(value.get("path", ""))).expanduser()
    if configured_path.resolve() != dataset.resolve():
        raise ValueError(f"{path}: path must resolve to {dataset}")
    if value.get("nc") != 1:
        raise ValueError(f"{path}: nc must be 1")
    names = value.get("names")
    valid_names = names == ["character"] or (
        isinstance(names, dict) and (names.get(0) == "character" or names.get("0") == "character")
    )
    if not valid_names:
        raise ValueError(f"{path}: names must define class 0 as 'character'")


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
