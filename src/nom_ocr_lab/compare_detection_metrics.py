"""Compare Week 3 and Week 4 metrics as Markdown and CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = (
    ("Precision", "metrics/precision(B)"),
    ("Recall", "metrics/recall(B)"),
    ("mAP50", "metrics/mAP50(B)"),
    ("mAP50-95", "metrics/mAP50-95(B)"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week3", type=Path, required=True)
    parser.add_argument("--week4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    before = json.loads(args.week3.read_text(encoding="utf-8"))["results"]
    after = json.loads(args.week4.read_text(encoding="utf-8"))["results"]
    rows = [
        (label, float(before[key]), float(after[key]), float(after[key]) - float(before[key]))
        for label, key in METRICS
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "week3", "week4", "delta"))
        writer.writerows(rows)
    markdown = ["| Metric | Week 3 | Week 4 | Delta |", "|---|---:|---:|---:|"]
    markdown.extend(
        f"| {label} | {before_value:.4f} | {after_value:.4f} | {delta:+.4f} |"
        for label, before_value, after_value, delta in rows
    )
    args.output.with_suffix(".md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
