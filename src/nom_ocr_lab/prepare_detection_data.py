"""Prepare single-class YOLO datasets for the Weeks 3–4 lab."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .detection import (
    prepare_mth_course_dataset,
    prepare_nomnaocr_archive,
    stage_mth_practice_sample,
    validate_mth_practice_dataset,
)


def _workspace_output(name: str) -> Path:
    workspace = os.environ.get("LAB_WORKSPACE")
    if not workspace:
        raise SystemExit("Set LAB_WORKSPACE or pass --output explicitly")
    return Path(workspace) / "datasets" / name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser(
        "stage-mth-practice", help="Copy the fixed raw-label Week 3 sample"
    )
    stage.add_argument("--source", type=Path, required=True)
    stage.add_argument("--output", type=Path)
    stage.add_argument("--seed", type=int, default=2026)

    validate = subparsers.add_parser(
        "validate-mth-practice", help="Validate the student-organized 18/2/4 dataset"
    )
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--source-manifest", type=Path)

    full = subparsers.add_parser(
        "prepare-mth-full", help="Prepare MTH1200 train/val and MTH1000 test"
    )
    full.add_argument("--train-source", type=Path, required=True)
    full.add_argument("--test-source", type=Path, required=True)
    full.add_argument("--output", type=Path)
    full.add_argument("--seed", type=int, default=2026)
    full.add_argument("--materialize", choices=("symlink", "copy"), default="symlink")
    full.add_argument(
        "--allow-shared-output",
        action="store_true",
        help=(
            "Allow staff to prepare only /data/shared/project2026/"
            "mth-full-project-ws2026"
        ),
    )

    nom = subparsers.add_parser(
        "prepare-nomnaocr-archive", help="Prepare the Roboflow ZIP as an external test set"
    )
    nom.add_argument("--archive", type=Path, required=True)
    nom.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "stage-mth-practice":
        output = args.output or _workspace_output("week03-source-sample")
        result = stage_mth_practice_sample(args.source, output, seed=args.seed)
    elif args.command == "validate-mth-practice":
        source_manifest = args.source_manifest or (
            _workspace_output("week03-source-sample") / "manifest.json"
        )
        result = validate_mth_practice_dataset(args.dataset, source_manifest)
        output = args.dataset
    elif args.command == "prepare-mth-full":
        output = args.output or _workspace_output("mth-course-full")
        result = prepare_mth_course_dataset(
            args.train_source,
            args.test_source,
            output,
            seed=args.seed,
            materialize=args.materialize,
            allow_shared_output=args.allow_shared_output,
        )
    elif args.command == "prepare-nomnaocr-archive":
        output = args.output or _workspace_output("nomnaocr-character-detection")
        result = prepare_nomnaocr_archive(args.archive, output)
    action = "Validated" if args.command == "validate-mth-practice" else "Prepared"
    print(f"{action} {result['size']} paired pages at {output.resolve()}")
    if "splits" in result:
        print(f"Splits: {result['splits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
