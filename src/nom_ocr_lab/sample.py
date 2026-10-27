"""Create and validate the persistent Week 1 NomNaOCR sample."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 2
READ_ONLY_SHARED_ROOT = Path("/data/shared/project2026")


@dataclass(frozen=True)
class PagePair:
    """An image and its matching line- and character-detection-label files."""

    image_path: Path
    label_path: Path
    detection_label_path: Path
    relative_image: Path
    relative_label: Path
    relative_detection_label: Path


def discover_page_pairs(dataset_root: Path) -> list[PagePair]:
    """Return complete page triplets in deterministic relative-path order."""

    dataset_root = dataset_root.resolve()
    image_root = dataset_root / "images"
    label_root = dataset_root / "line_labels"
    detection_label_root = dataset_root / "detection_labels"
    if (
        not image_root.is_dir()
        or not label_root.is_dir()
        or not detection_label_root.is_dir()
    ):
        raise FileNotFoundError(
            "Expected images/, line_labels/, and detection_labels/ directories "
            f"under {dataset_root}"
        )

    pairs = []
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative_image = image_path.relative_to(image_root)
        relative_label = relative_image.with_suffix(".txt")
        relative_detection_label = relative_label
        label_path = label_root / relative_label
        detection_label_path = detection_label_root / relative_detection_label
        if label_path.is_file() and detection_label_path.is_file():
            pairs.append(
                PagePair(
                    image_path,
                    label_path,
                    detection_label_path,
                    relative_image,
                    relative_label,
                    relative_detection_label,
                )
            )
    return pairs


def prepare_sample(source: Path, destination: Path, size: int = 5) -> tuple[list[PagePair], bool]:
    """Create or validate a deterministic sample of complete page triplets.

    Returns ``(pairs, created)``. A valid existing sample is reused. Any partial,
    changed, or conflicting destination raises an error instead of being overwritten.
    """

    source = source.resolve()
    destination = destination.resolve()
    if size < 1:
        raise ValueError("size must be at least 1")
    shared_root = READ_ONLY_SHARED_ROOT.resolve()
    if destination == shared_root or shared_root in destination.parents:
        raise ValueError(f"The destination must not be inside read-only {shared_root}")
    if destination == source or source in destination.parents:
        raise ValueError("The destination must not be inside the source dataset")

    manifest_path = destination / MANIFEST_NAME
    if destination.exists() and any(destination.iterdir()):
        if not manifest_path.is_file():
            raise FileExistsError(
                f"Refusing to overwrite non-empty destination without {MANIFEST_NAME}: "
                f"{destination}"
            )
        return _validate_existing_sample(source, destination, size), False

    source_pairs = discover_page_pairs(source)
    if len(source_pairs) < size:
        noun = "triplet" if len(source_pairs) == 1 else "triplets"
        raise RuntimeError(
            f"Found only {len(source_pairs)} complete page {noun}; requested {size}"
        )
    selected = source_pairs[:size]

    copied_entries = []
    for pair in selected:
        destination_image = destination / "images" / pair.relative_image
        destination_label = destination / "line_labels" / pair.relative_label
        destination_detection_label = (
            destination / "detection_labels" / pair.relative_detection_label
        )
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        destination_detection_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pair.image_path, destination_image)
        shutil.copy2(pair.label_path, destination_label)
        shutil.copy2(pair.detection_label_path, destination_detection_label)
        copied_entries.append(
            {
                "image": pair.relative_image.as_posix(),
                "image_sha256": _sha256(destination_image),
                "line_label": pair.relative_label.as_posix(),
                "line_label_sha256": _sha256(destination_label),
                "detection_label": pair.relative_detection_label.as_posix(),
                "detection_label_sha256": _sha256(destination_detection_label),
            }
        )

    manifest = {
        "version": MANIFEST_VERSION,
        "source": str(source),
        "size": size,
        "pairs": copied_entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return discover_page_pairs(destination), True


def _validate_existing_sample(source: Path, destination: Path, size: int) -> list[PagePair]:
    manifest_path = destination / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid sample manifest: {manifest_path}") from exc

    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported sample manifest version in {manifest_path}")
    if manifest.get("source") != str(source) or manifest.get("size") != size:
        raise ValueError("Existing sample was created from different source or size settings")

    entries = manifest.get("pairs")
    if not isinstance(entries, list) or len(entries) != size:
        raise ValueError("Existing sample manifest has the wrong number of page triplets")

    for entry in entries:
        try:
            relative_image = _validated_relative_path(entry["image"])
            relative_label = _validated_relative_path(entry["line_label"])
            relative_detection_label = _validated_relative_path(
                entry["detection_label"]
            )
            image_path = destination / "images" / relative_image
            label_path = destination / "line_labels" / relative_label
            detection_label_path = (
                destination / "detection_labels" / relative_detection_label
            )
            expected_image_hash = entry["image_sha256"]
            expected_label_hash = entry["line_label_sha256"]
            expected_detection_label_hash = entry["detection_label_sha256"]
        except (KeyError, TypeError) as exc:
            raise ValueError("Existing sample manifest contains a malformed entry") from exc
        if (
            not image_path.is_file()
            or not label_path.is_file()
            or not detection_label_path.is_file()
        ):
            raise ValueError("Existing sample is incomplete; a recorded file is missing")
        if (
            _sha256(image_path) != expected_image_hash
            or _sha256(label_path) != expected_label_hash
            or _sha256(detection_label_path) != expected_detection_label_hash
        ):
            raise ValueError("Existing sample differs from its manifest")

    pairs = discover_page_pairs(destination)
    if len(pairs) != size:
        raise ValueError("Existing sample contains unexpected or unmatched files")
    return pairs


def _validated_relative_path(value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError("Manifest paths must be strings")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Manifest paths must stay inside the sample dataset")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
