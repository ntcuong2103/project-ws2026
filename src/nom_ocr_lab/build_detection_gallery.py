"""Build character-level ground-truth/prediction overlays and an error summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detection_visualization import create_error_gallery


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--selection", choices=("all", "errors"), default="errors")
    args = parser.parse_args(argv)
    report = create_error_gallery(
        args.dataset,
        args.predictions,
        args.output,
        split=args.split,
        iou_threshold=args.iou,
        limit=args.limit,
        selection=args.selection,
    )
    print(json.dumps(report["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
