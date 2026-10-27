"""Download and verify the pinned IDS resources used by the recognizer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .recognition import prepare_resources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output
    if output is None:
        workspace = os.environ.get("LAB_WORKSPACE")
        if not workspace:
            raise SystemExit("Set LAB_WORKSPACE or pass --output")
        output = Path(workspace) / "datasets" / "ids"
    manifest = prepare_resources(output)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
