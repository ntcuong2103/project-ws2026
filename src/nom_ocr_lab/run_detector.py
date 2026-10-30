"""Train, evaluate, or predict with the course's single-class detector."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _default_artifact_root(kind: str) -> Path:
    workspace = os.environ.get("LAB_WORKSPACE")
    if not workspace:
        raise SystemExit("Set LAB_WORKSPACE or pass --output-root explicitly")
    return Path(workspace) / kind


def _refuse_nonempty_run(output_root: Path, name: str) -> Path:
    run_root = output_root / name
    if run_root.exists() and any(run_root.iterdir()):
        raise SystemExit(f"Refusing to overwrite non-empty run: {run_root}")
    return run_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--model", default="yolo11n.pt")
    train.add_argument("--name", required=True)
    train.add_argument("--output-root", type=Path)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=1280)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--patience", type=int, default=10)
    train.add_argument("--device", default="0")
    train.add_argument("--max-det", type=int, default=1000)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--optimizer", default="auto")
    train.add_argument("--lr0", type=float, default=0.01)
    train.add_argument("--mosaic", type=float, default=1.0)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluate.add_argument("--name", required=True)
    evaluate.add_argument("--output-root", type=Path)
    evaluate.add_argument("--imgsz", type=int, default=1280)
    evaluate.add_argument("--batch", type=int, default=8)
    evaluate.add_argument("--device", default="0")
    evaluate.add_argument("--max-det", type=int, default=1000)

    predict = commands.add_parser("predict")
    predict.add_argument("--source", type=Path, required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--name", required=True)
    predict.add_argument("--output-root", type=Path)
    predict.add_argument("--imgsz", type=int, default=1280)
    predict.add_argument("--batch", type=int, default=8)
    predict.add_argument("--device", default="0")
    predict.add_argument("--conf", type=float, default=0.25)
    predict.add_argument("--iou", type=float, default=0.5)
    predict.add_argument("--max-det", type=int, default=1000)
    return parser


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _prediction_manifest(
    *,
    source_root: Path,
    model: Path,
    args: argparse.Namespace,
    images: list[Path],
) -> dict[str, object]:
    """Describe a complete detector handoff, including pages without boxes."""

    return {
        "version": 1,
        "kind": "detector-predictions",
        "source_images": str(source_root),
        "model": str(model.resolve()),
        "settings": {
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": str(args.device),
            "confidence": args.conf,
            "nms_iou": args.iou,
            "max_det": args.max_det,
            "single_cls": True,
        },
        "pages": [
            {
                "page_id": relative.with_suffix("").as_posix(),
                "image": relative.as_posix(),
                "labels": relative.with_suffix(".txt").as_posix(),
            }
            for relative in images
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install lab dependencies with `uv sync` before running YOLO") from exc

    if args.command == "train":
        output_root = args.output_root or _default_artifact_root("models")
        run_root = _refuse_nonempty_run(output_root, args.name)
        model = YOLO(args.model)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=args.device,
            single_cls=True,
            project=str(output_root),
            name=args.name,
            save=True,
            max_det=args.max_det,
            seed=args.seed,
            optimizer=args.optimizer,
            lr0=args.lr0,
            mosaic=args.mosaic,
            plots=True,
        )
        missing = [
            path.relative_to(run_root).as_posix()
            for path in (run_root / "weights/best.pt", run_root / "weights/last.pt")
            if not path.is_file()
        ]
        if missing:
            raise SystemExit(f"Training finished without required artifacts: {missing}")
        return 0

    if args.command == "evaluate":
        output_root = args.output_root or _default_artifact_root("metrics")
        result_dir = _refuse_nonempty_run(output_root, args.name)
        model = YOLO(str(args.model))
        metrics = model.val(
            data=str(args.data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            single_cls=True,
            project=str(output_root),
            name=args.name,
            plots=True,
            max_det=args.max_det,
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(args.model.resolve()),
            "data": str(args.data.resolve()),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "max_det": args.max_det,
            "single_cls": True,
            "results": _jsonable(metrics.results_dict),
        }
        (result_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
        return 0

    output_root = args.output_root or _default_artifact_root("predictions")
    run_root = _refuse_nonempty_run(output_root, args.name)
    source_root = args.source.resolve()
    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    source_images = (
        [path for path in sorted(source_root.rglob("*")) if path.is_file() and path.suffix.lower() in image_suffixes]
        if source_root.is_dir()
        else [source_root]
    )
    prediction_source = [str(path) for path in source_images] if source_root.is_dir() else str(source_root)
    if isinstance(prediction_source, list) and not prediction_source:
        raise SystemExit(f"No supported images found below {source_root}")
    model = YOLO(str(args.model))
    results = model.predict(
        source=prediction_source,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=False,
        save_txt=False,
        single_cls=True,
        stream=True,
    )
    for result in results:
        # Do not resolve individual image symlinks: their path below the prepared
        # dataset carries the document-relative identity we need to preserve.
        result_path = Path(result.path).absolute()
        relative = (
            result_path.relative_to(source_root)
            if source_root.is_dir()
            else Path(result_path.name)
        )
        image_output = run_root / "images" / relative
        label_output = run_root / "labels" / relative.with_suffix(".txt")
        image_output.parent.mkdir(parents=True, exist_ok=True)
        label_output.parent.mkdir(parents=True, exist_ok=True)
        result.save(filename=str(image_output))
        result.save_txt(str(label_output), save_conf=True)
        # Ultralytics may omit an empty label file. The recognition handoff must
        # distinguish a page with no boxes from a page that was never processed.
        label_output.touch(exist_ok=True)
    relative_images = [
        path.relative_to(source_root) if source_root.is_dir() else Path(path.name)
        for path in source_images
    ]
    payload = _prediction_manifest(
        source_root=source_root,
        model=args.model,
        args=args,
        images=relative_images,
    )
    (run_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
