"""Parsing helpers for NomNaOCR line and character-position annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LineAnnotation:
    """One quadrilateral line region and its transcription."""

    points: tuple[tuple[float, float], ...]
    text: str


@dataclass(frozen=True)
class DetectionAnnotation:
    """One normalized YOLO character-position box without a character identity."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_line_annotation(raw_line: str) -> LineAnnotation:
    """Parse ``x1,y1,...,x4,y4,text`` into visualization-ready data."""

    parts = raw_line.strip().split(",", maxsplit=8)
    if len(parts) != 9:
        raise ValueError(f"Expected 8 coordinates and text, got {len(parts)} fields")

    try:
        coordinates = tuple(float(value) for value in parts[:8])
    except ValueError as exc:
        raise ValueError("Line annotation contains a non-numeric coordinate") from exc

    points = tuple(zip(coordinates[0::2], coordinates[1::2]))
    return LineAnnotation(points=points, text=parts[8])


def read_line_annotations(path: Path) -> list[LineAnnotation]:
    """Read all non-empty annotations from a UTF-8 label file."""

    annotations = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            annotations.append(parse_line_annotation(raw_line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return annotations


def parse_detection_annotation(raw_line: str) -> DetectionAnnotation:
    """Parse a normalized ``class x_center y_center width height`` YOLO row."""

    parts = raw_line.split()
    if len(parts) != 5:
        raise ValueError(f"Expected 5 YOLO fields, got {len(parts)}")
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = (float(value) for value in parts[1:])
    except ValueError as exc:
        raise ValueError("Detection annotation contains a non-numeric field") from exc
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        raise ValueError("Detection center coordinates must be in [0, 1]")
    if not (0 < width <= 1 and 0 < height <= 1):
        raise ValueError("Detection width and height must be in (0, 1]")
    if (
        x_center - width / 2 < 0
        or x_center + width / 2 > 1
        or y_center - height / 2 < 0
        or y_center + height / 2 > 1
    ):
        raise ValueError("Detection box extends outside the normalized image bounds")
    return DetectionAnnotation(class_id, x_center, y_center, width, height)


def read_detection_annotations(path: Path) -> list[DetectionAnnotation]:
    """Read all non-empty character-position annotations from a YOLO label file."""

    annotations = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        try:
            annotations.append(parse_detection_annotation(raw_line))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return annotations
