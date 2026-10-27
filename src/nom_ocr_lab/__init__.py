"""Shared utilities for the Nôm OCR course labs."""

from .annotations import LineAnnotation, parse_line_annotation, read_line_annotations
from .sample import PagePair, discover_page_pairs, prepare_sample

__all__ = [
    "LineAnnotation",
    "PagePair",
    "discover_page_pairs",
    "parse_line_annotation",
    "prepare_sample",
    "read_line_annotations",
]

