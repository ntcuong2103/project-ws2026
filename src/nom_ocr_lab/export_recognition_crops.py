"""Export one detector prediction page as unlabeled recognizer crops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .recognition_crops import export_crops


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expand-ratio", type=float, default=1.2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = export_crops(
        args.image,
        args.labels,
        args.output,
        expand_ratio=args.expand_ratio,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
