"""Export detector boxes as unlabeled character crops for recognizer inference."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class DetectionPrediction:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float | None = None


def parse_prediction(raw: str) -> DetectionPrediction:
    """Parse a five-field YOLO box or a six-field prediction with confidence."""

    fields = raw.split()
    if len(fields) not in (5, 6):
        raise ValueError(f"Expected 5 or 6 YOLO fields, got {len(fields)}")
    try:
        class_id = int(fields[0])
        values = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError("Prediction fields must be numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Prediction fields must be finite")
    x_center, y_center, width, height = values[:4]
    confidence = values[4] if len(values) == 5 else None
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        raise ValueError("Prediction center coordinates must be in [0, 1]")
    if not (0 < width <= 1 and 0 < height <= 1):
        raise ValueError("Prediction width and height must be in (0, 1]")
    if confidence is not None and not 0 <= confidence <= 1:
        raise ValueError("Prediction confidence must be in [0, 1]")
    if x_center - width / 2 < 0 or x_center + width / 2 > 1:
        raise ValueError("Prediction box extends outside image width")
    if y_center - height / 2 < 0 or y_center + height / 2 > 1:
        raise ValueError("Prediction box extends outside image height")
    return DetectionPrediction(
        class_id, x_center, y_center, width, height, confidence
    )


def read_predictions(path: Path) -> list[DetectionPrediction]:
    predictions = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            predictions.append(parse_prediction(raw))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return predictions


def expanded_crop_box(
    prediction: DetectionPrediction,
    *,
    image_width: int,
    image_height: int,
    expand_ratio: float = 1.2,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Match the source recognizer's integer conversion and square expansion."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if expand_ratio <= 0:
        raise ValueError("expand_ratio must be positive")
    x = int(prediction.x_center * image_width)
    y = int(prediction.y_center * image_height)
    width = int(prediction.width * image_width)
    height = int(prediction.height * image_height)
    if width < 1 or height < 1:
        raise ValueError("Prediction becomes an empty pixel box")
    pixel_box = (
        int(x - width / 2),
        int(y - height / 2),
        int(x + width / 2),
        int(y + height / 2),
    )
    side = max(width, height) * expand_ratio
    expanded = (
        max(0, int(x - side / 2)),
        max(0, int(y - side / 2)),
        min(image_width, int(x + side / 2)),
        min(image_height, int(y + side / 2)),
    )
    if expanded[0] >= expanded[2] or expanded[1] >= expanded[3]:
        raise ValueError("Expanded crop is empty")
    return pixel_box, expanded


def export_crops(
    image_path: Path,
    labels_path: Path,
    output: Path,
    *,
    expand_ratio: float = 1.2,
) -> dict[str, object]:
    """Save RGB PNG crops plus JSON and TSV manifests without assigning identities."""

    image_path = image_path.resolve()
    labels_path = labels_path.resolve()
    predictions = read_predictions(labels_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    calculated = [
        expanded_crop_box(
            prediction,
            image_width=image.width,
            image_height=image.height,
            expand_ratio=expand_ratio,
        )
        for prediction in predictions
    ]

    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty crop directory: {output}")
    crop_root = output / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)

    entries = []
    for box_id, (prediction, (pixel_box, expanded_box)) in enumerate(
        zip(predictions, calculated)
    ):
        crop_name = f"box-{box_id:04d}.png"
        image.crop(expanded_box).save(crop_root / crop_name)
        entries.append(
            {
                "box_id": box_id,
                "class_id": prediction.class_id,
                "confidence": prediction.confidence,
                "normalized_box": [
                    prediction.x_center,
                    prediction.y_center,
                    prediction.width,
                    prediction.height,
                ],
                "pixel_box": list(pixel_box),
                "expanded_box": list(expanded_box),
                "crop": f"crops/{crop_name}",
            }
        )

    manifest = {
        "version": 1,
        "kind": "unlabeled-recognizer-inference-crops",
        "source_image": str(image_path),
        "source_labels": str(labels_path),
        "image_size": [image.width, image.height],
        "expand_ratio": expand_ratio,
        "labeled": False,
        "count": len(entries),
        "entries": entries,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "box_id",
        "class_id",
        "confidence",
        "normalized_box",
        "pixel_box",
        "expanded_box",
        "crop",
    ]
    with (output / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for entry in entries:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, list)
                    else value
                    for key, value in entry.items()
                }
            )
    return manifest
