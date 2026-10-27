"""Shared utilities for the Nôm OCR course labs."""

from .annotations import (
    DetectionAnnotation,
    LineAnnotation,
    parse_detection_annotation,
    parse_line_annotation,
    read_detection_annotations,
    read_line_annotations,
)
from .sample import PagePair, discover_page_pairs, prepare_sample

__all__ = [
    "DetectionAnnotation",
    "LineAnnotation",
    "PagePair",
    "discover_page_pairs",
    "parse_detection_annotation",
    "parse_line_annotation",
    "prepare_sample",
    "read_detection_annotations",
    "read_line_annotations",
]
