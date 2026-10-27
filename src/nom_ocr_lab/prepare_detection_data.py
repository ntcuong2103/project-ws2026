"""Prepare single-class YOLO datasets for the Weeks 3–4 lab."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .detection import prepare_mth1000, prepare_yolo_view


def _workspace_output(name: str) -> Path:
    workspace = os.environ.get("LAB_WORKSPACE")
    if not workspace:
        raise SystemExit("Set LAB_WORKSPACE or pass --output explicitly")
    return Path(workspace) / "datasets" / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    mth = subparsers.add_parser("mth1000", help="Convert MTH1000 label_char data")
    mth.add_argument("--source", type=Path, required=True)
    mth.add_argument("--output", type=Path)
    mth.add_argument("--mode", choices=("practice", "full"), required=True)
    mth.add_argument("--sample-size", type=int, default=24)
    mth.add_argument("--seed", type=int, default=2026)
    mth.add_argument("--materialize", choices=("symlink", "copy"), default="symlink")

    nom = subparsers.add_parser("nomnaocr", help="Build a YOLO view of NomNaOCR")
    nom.add_argument("--source", type=Path, required=True)
    nom.add_argument("--output", type=Path)
    nom.add_argument("--materialize", choices=("symlink", "copy"), default="symlink")

    nakagawa = subparsers.add_parser("nakagawa", help="Build a split YOLO view of Nakagawa")
    nakagawa.add_argument("--source", type=Path, required=True)
    nakagawa.add_argument("--output", type=Path)
    nakagawa.add_argument("--materialize", choices=("symlink", "copy"), default="symlink")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "mth1000":
        output = args.output or _workspace_output(f"mth1000-{args.mode}")
        manifest = prepare_mth1000(
            args.source,
            output,
            mode=args.mode,
            sample_size=args.sample_size,
            seed=args.seed,
            materialize=args.materialize,
        )
    elif args.command == "nomnaocr":
        output = args.output or _workspace_output("nomnaocr-character-detection")
        manifest = prepare_yolo_view(
            args.source,
            output,
            label_dir="detection_labels",
            materialize=args.materialize,
        )
    else:
        output = args.output or _workspace_output("nakagawa-character-detection")
        split_files = {
            split: args.source / f"{split}.txt" for split in ("train", "val", "test")
        }
        manifest = prepare_yolo_view(
            args.source,
            output,
            label_dir="labels",
            split_files=split_files,
            materialize=args.materialize,
        )
    print(f"Prepared {manifest['size']} paired pages at {output.resolve()}")
    print(f"Splits: {manifest['splits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
