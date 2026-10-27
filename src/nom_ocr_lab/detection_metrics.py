"""Dependency-light scoring for saved single-class YOLO detections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .detection import IMAGE_SUFFIXES
from .detection_visualization import Box, box_iou, read_yolo_boxes


@dataclass(frozen=True)
class SavedPrediction:
    page: str
    box: Box


def parse_iou_thresholds(value: str) -> tuple[float, ...]:
    """Parse a comma-separated, increasing list of IoU thresholds."""

    try:
        thresholds = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("IoU thresholds must be comma-separated numbers") from exc
    if not thresholds or any(not 0 < item <= 1 for item in thresholds):
        raise ValueError("IoU thresholds must be in (0, 1]")
    if tuple(sorted(set(thresholds))) != thresholds:
        raise ValueError("IoU thresholds must be unique and increasing")
    return thresholds


def _match_at_iou(
    ground_truth: dict[str, list[Box]],
    predictions: Iterable[SavedPrediction],
    threshold: float,
) -> tuple[list[int], list[int]]:
    matched = {page: set() for page in ground_truth}
    true_positives: list[int] = []
    false_positives: list[int] = []
    ordered = sorted(predictions, key=lambda item: item.box.confidence, reverse=True)
    for prediction in ordered:
        candidates = [
            (box_iou(prediction.box, truth), index)
            for index, truth in enumerate(ground_truth[prediction.page])
            if index not in matched[prediction.page]
        ]
        overlap, truth_index = max(candidates, default=(0.0, -1))
        is_match = overlap >= threshold
        if is_match:
            matched[prediction.page].add(truth_index)
        true_positives.append(int(is_match))
        false_positives.append(int(not is_match))
    return true_positives, false_positives


def _precision_recall(
    true_positives: list[int], false_positives: list[int], ground_truth_count: int
) -> tuple[list[float], list[float]]:
    precision = []
    recall = []
    tp_total = 0
    fp_total = 0
    for tp, fp in zip(true_positives, false_positives, strict=True):
        tp_total += tp
        fp_total += fp
        precision.append(tp_total / (tp_total + fp_total))
        recall.append(tp_total / ground_truth_count if ground_truth_count else 0.0)
    return precision, recall


def interpolated_average_precision(
    precision: list[float], recall: list[float]
) -> float:
    """Return 101-point interpolated AP, the convention used by COCO metrics."""

    if not precision:
        return 0.0
    samples = []
    for index in range(101):
        recall_level = index / 100
        samples.append(
            max((p for p, r in zip(precision, recall, strict=True) if r >= recall_level), default=0.0)
        )
    return sum(samples) / len(samples)


def evaluate_saved_predictions(
    dataset_root: Path,
    prediction_root: Path,
    *,
    split: str = "test",
    iou_thresholds: tuple[float, ...] = tuple(index / 100 for index in range(50, 100, 5)),
) -> dict[str, object]:
    """Evaluate nested saved predictions against a prepared YOLO dataset."""

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if (
        not iou_thresholds
        or any(not 0 < item <= 1 for item in iou_thresholds)
        or tuple(sorted(set(iou_thresholds))) != iou_thresholds
    ):
        raise ValueError("IoU thresholds must be unique, increasing, and in (0, 1]")
    image_root = dataset_root / "images" / split
    label_root = dataset_root / "labels" / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"Missing images/{split} or labels/{split} below {dataset_root}")

    page_paths = [
        path.relative_to(image_root)
        for path in sorted(image_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not page_paths:
        raise ValueError(f"No supported images below {image_root}")

    ground_truth: dict[str, list[Box]] = {}
    predictions: list[SavedPrediction] = []
    expected_prediction_paths = set()
    for relative_image in page_paths:
        page = relative_image.with_suffix("").as_posix()
        truth_path = label_root / relative_image.with_suffix(".txt")
        if not truth_path.is_file():
            raise FileNotFoundError(f"Missing ground-truth label for {relative_image}")
        ground_truth[page] = read_yolo_boxes(truth_path)
        prediction_path = prediction_root / relative_image.with_suffix(".txt")
        expected_prediction_paths.add(prediction_path.resolve())
        for box in read_yolo_boxes(prediction_path, require_confidence=True):
            predictions.append(SavedPrediction(page, box))

    extra_predictions = (
        [
            path
            for path in prediction_root.rglob("*.txt")
            if path.resolve() not in expected_prediction_paths
        ]
        if prediction_root.is_dir()
        else []
    )
    if extra_predictions:
        raise ValueError(f"Prediction has no matching dataset image: {extra_predictions[0]}")

    ground_truth_count = sum(len(boxes) for boxes in ground_truth.values())
    if ground_truth_count == 0:
        raise ValueError("Evaluation requires at least one ground-truth box")

    per_iou: dict[str, dict[str, float | int]] = {}
    for threshold in iou_thresholds:
        tp_flags, fp_flags = _match_at_iou(ground_truth, predictions, threshold)
        precision_curve, recall_curve = _precision_recall(
            tp_flags, fp_flags, ground_truth_count
        )
        tp = sum(tp_flags)
        fp = sum(fp_flags)
        fn = ground_truth_count - tp
        per_iou[f"{threshold:.2f}"] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / ground_truth_count,
            "ap": interpolated_average_precision(precision_curve, recall_curve),
        }

    ap50_key = next((key for key in per_iou if abs(float(key) - 0.5) < 1e-9), None)
    if ap50_key is None:
        raise ValueError("IoU thresholds must include 0.50 to report AP50")
    ap50_metrics = per_iou[ap50_key]
    return {
        "dataset": str(dataset_root.resolve()),
        "predictions": str(prediction_root.resolve()),
        "split": split,
        "pages": len(page_paths),
        "ground_truth_boxes": ground_truth_count,
        "prediction_boxes": len(predictions),
        "iou_thresholds": list(iou_thresholds),
        "tp": ap50_metrics["tp"],
        "fp": ap50_metrics["fp"],
        "fn": ap50_metrics["fn"],
        "precision": ap50_metrics["precision"],
        "recall": ap50_metrics["recall"],
        "ap50": ap50_metrics["ap"],
        "map50_95": sum(float(item["ap"]) for item in per_iou.values()) / len(per_iou),
        "per_iou": per_iou,
    }


def metrics_markdown(report: dict[str, object]) -> str:
    """Render the saved-label evaluation summary as Markdown."""

    lines = [
        "# Saved-label detection metrics",
        "",
        f"- Split: `{report['split']}`",
        f"- Pages: {report['pages']}",
        f"- Ground-truth boxes: {report['ground_truth_boxes']}",
        f"- Prediction boxes: {report['prediction_boxes']}",
        "",
        "| IoU | TP | FP | FN | Precision | Recall | AP |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold, values in report["per_iou"].items():
        lines.append(
            f"| {threshold} | {values['tp']} | {values['fp']} | {values['fn']} | "
            f"{values['precision']:.4f} | {values['recall']:.4f} | {values['ap']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"**AP50:** {report['ap50']:.4f}",
            "",
            f"**mAP50–95:** {report['map50_95']:.4f}",
            "",
        ]
    )
    return "\n".join(lines)
