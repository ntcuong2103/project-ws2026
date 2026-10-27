"""Parsing helpers for NomNaOCR line-level annotations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LineAnnotation:
    """One quadrilateral line region and its transcription."""

    points: tuple[tuple[float, float], ...]
    text: str


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

