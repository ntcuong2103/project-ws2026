"""Evaluate saved, confidence-bearing YOLO labels without a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detection import READ_ONLY_SHARED_ROOT
from .detection_metrics import (
    evaluate_saved_predictions,
    metrics_markdown,
    parse_iou_thresholds,
)


DEFAULT_THRESHOLDS = "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iou-thresholds", default=DEFAULT_THRESHOLDS)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    shared = READ_ONLY_SHARED_ROOT.resolve()
    if output == shared or shared in output.parents:
        raise SystemExit(f"Output must not be inside read-only {shared}")
    if output.exists() and not output.is_dir():
        raise SystemExit(f"Output exists and is not a directory: {output}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty output: {output}")
    try:
        thresholds = parse_iou_thresholds(args.iou_thresholds)
        report = evaluate_saved_predictions(
            args.dataset,
            args.predictions,
            split=args.split,
            iou_thresholds=thresholds,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "metrics.md").write_text(metrics_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
