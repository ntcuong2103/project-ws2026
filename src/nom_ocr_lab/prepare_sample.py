"""Command-line entry point for the Week 1 sample."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .sample import prepare_sample


DEFAULT_SOURCE = Path("/data/shared/project2026/nomnaocr")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or validate the persistent Week 1 NomNaOCR sample."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=os.environ.get("LAB_WORKSPACE"),
        help="Lab workspace root; defaults to LAB_WORKSPACE",
    )
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workspace is None:
        raise SystemExit("Set LAB_WORKSPACE or pass --workspace")
    destination = args.workspace / "datasets" / "nomnaocr-sample"
    pairs, created = prepare_sample(args.source, destination, args.size)
    action = "Created" if created else "Reused"
    print(f"{action} {len(pairs)} page triplets at {destination.resolve()}")
    for pair in pairs:
        print(pair.relative_image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
