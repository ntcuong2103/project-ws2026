"""Dependency-light overlays and error summaries for YOLO character boxes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .detection import IMAGE_SUFFIXES


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


def read_yolo_boxes(path: Path, *, require_confidence: bool = False) -> list[Box]:
    """Read and validate normalized single-class YOLO labels."""

    if not path.is_file():
        return []
    boxes = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        expected = "6" if require_confidence else "5 or 6"
        if len(fields) not in ((6,) if require_confidence else (5, 6)):
            raise ValueError(f"{path}:{line_number}: expected {expected} fields")
        try:
            values = [float(value) for value in fields]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: non-numeric field") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: fields must be finite")
        class_id, x, y, width, height, *confidence = values
        if class_id != 0:
            raise ValueError(f"{path}:{line_number}: expected single class 0")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(
                f"{path}:{line_number}: normalized coordinates must be in [0, 1] "
                "with positive width and height"
            )
        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: box exceeds image width")
        if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            raise ValueError(f"{path}:{line_number}: box exceeds image height")
        if confidence and not 0 <= confidence[0] <= 1:
            raise ValueError(f"{path}:{line_number}: confidence must be in [0, 1]")
        boxes.append(
            Box(
                x - width / 2,
                y - height / 2,
                x + width / 2,
                y + height / 2,
                confidence[0] if confidence else 1.0,
            )
        )
    return boxes


def box_iou(first: Box, second: Box) -> float:
    width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = width * height
    first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
    second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def summarize_errors(
    ground_truth: list[Box], predictions: list[Box], *, iou_threshold: float = 0.5
) -> dict[str, int]:
    """Summarize character-level localization errors for one page."""

    candidates = sorted(
        (
            (box_iou(prediction, truth), pred_index, truth_index)
            for pred_index, prediction in enumerate(predictions)
            for truth_index, truth in enumerate(ground_truth)
        ),
        reverse=True,
    )
    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    for overlap, pred_index, truth_index in candidates:
        if overlap < iou_threshold:
            break
        if pred_index not in matched_predictions and truth_index not in matched_truth:
            matched_predictions.add(pred_index)
            matched_truth.add(truth_index)

    unmatched_predictions = set(range(len(predictions))) - matched_predictions
    max_unmatched_iou = {
        index: max((box_iou(predictions[index], truth) for truth in ground_truth), default=0.0)
        for index in unmatched_predictions
    }
    overlap_floor = 0.1
    merges = sum(
        sum(box_iou(prediction, truth) >= overlap_floor for truth in ground_truth) >= 2
        for prediction in predictions
    )
    splits = sum(
        sum(box_iou(prediction, truth) >= overlap_floor for prediction in predictions) >= 2
        for truth in ground_truth
    )
    return {
        "matched": len(matched_truth),
        "missed": len(ground_truth) - len(matched_truth),
        "spurious": sum(value < overlap_floor for value in max_unmatched_iou.values()),
        "poor_localization": sum(
            overlap_floor <= value < iou_threshold for value in max_unmatched_iou.values()
        ),
        "merge_candidates": merges,
        "split_candidates": splits,
    }


def draw_overlay(
    image_path: Path,
    ground_truth: list[Box],
    predictions: list[Box],
    output_path: Path,
) -> None:
    """Draw ground truth in green and predictions in red."""

    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in ground_truth:
        draw.rectangle(
            (box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height),
            outline="lime",
            width=2,
        )
    for box in predictions:
        draw.rectangle(
            (box.x1 * width, box.y1 * height, box.x2 * width, box.y2 * height),
            outline="red",
            width=2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def create_error_gallery(
    dataset_root: Path,
    prediction_root: Path,
    output_root: Path,
    *,
    split: str = "test",
    iou_threshold: float = 0.5,
    limit: int = 10,
    selection: str = "errors",
) -> dict[str, object]:
    """Create overlays for all pages or only pages containing detection errors."""

    if selection not in {"all", "errors"}:
        raise ValueError("selection must be 'all' or 'errors'")
    if limit <= 0:
        raise ValueError("limit must be positive")

    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    totals = {
        "matched": 0,
        "missed": 0,
        "spurious": 0,
        "poor_localization": 0,
        "merge_candidates": 0,
        "split_candidates": 0,
    }
    pages = []
    for image_path in sorted(image_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = image_path.relative_to(image_root)
        truth = read_yolo_boxes(label_root / relative.with_suffix(".txt"))
        predicted = read_yolo_boxes(prediction_root / relative.with_suffix(".txt"))
        summary = summarize_errors(truth, predicted, iou_threshold=iou_threshold)
        for key, value in summary.items():
            totals[key] += value
        has_errors = any(summary[key] for key in summary if key != "matched")
        if (selection == "all" or has_errors) and len(pages) < limit:
            output_path = output_root / relative.with_suffix(".jpg")
            draw_overlay(image_path, truth, predicted, output_path)
            pages.append({"image": relative.as_posix(), **summary})

    report: dict[str, object] = {
        "iou_threshold": iou_threshold,
        "selection": selection,
        "legend": {"ground_truth": "green", "prediction": "red"},
        "totals": totals,
        "gallery_pages": pages,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "error_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
