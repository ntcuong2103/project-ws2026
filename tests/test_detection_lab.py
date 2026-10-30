from __future__ import annotations

import json
import shutil
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from nom_ocr_lab.detection import (
    convert_xyxy_label,
    normalize_yolo_label,
    prepare_mth_course_dataset,
    prepare_mth1000,
    prepare_nomnaocr_archive,
    prepare_yolo_view,
    split_pairs,
    stage_mth_practice_sample,
    validate_mth_practice_dataset,
)
from nom_ocr_lab.detection_visualization import (
    Box,
    box_iou,
    create_error_gallery,
    read_yolo_boxes,
    summarize_errors,
)
from nom_ocr_lab.detection_metrics import (
    evaluate_saved_predictions,
    interpolated_average_precision,
    parse_iou_thresholds,
)
from nom_ocr_lab.evaluate_detection_predictions import main as evaluate_predictions_main
from nom_ocr_lab.run_detector import build_parser as build_detector_parser


def make_mth1000(root: Path, count: int = 30) -> Path:
    for index in range(count):
        relative = Path(f"volume-{index % 3}") / f"page-{index:03d}"
        image_path = root / "images" / relative.with_suffix(".jpg")
        label_path = root / "label_char" / relative.with_suffix(".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 200), "white").save(image_path)
        label_path.write_text("17 10 20 30 60\n42 50 80 90 160\n", encoding="utf-8")
    return root


def make_yolo_source(
    root: Path, names: list[str], *, label_dir: str = "detection_labels"
) -> Path:
    for name in names:
        image_path = root / "images" / f"{name}.jpg"
        label_path = root / label_dir / f"{name}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 60), "white").save(image_path)
        label_path.write_text("7503 0.500000 0.500000 0.200000 0.300000\n", encoding="utf-8")
    return root


def all_label_rows_start_with_zero(root: Path) -> bool:
    return all(
        row.startswith("0 ")
        for path in root.rglob("*.txt")
        for row in path.read_text(encoding="utf-8").splitlines()
        if row
    )


def test_pixel_conversion_forces_single_class(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("99 10 20 30 60\n", encoding="utf-8")

    converted = convert_xyxy_label(label, image_width=100, image_height=200)

    assert converted == "0 0.200000 0.200000 0.200000 0.200000\n"

    _, x, y, width, height = map(float, converted.split())
    assert ((x - width / 2) * 100, (y - height / 2) * 200) == pytest.approx(
        (10, 20)
    )
    assert ((x + width / 2) * 100, (y + height / 2) * 200) == pytest.approx(
        (30, 60)
    )


def test_pixel_conversion_accepts_character_glyph_labels(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("經 10 20 30 60\n", encoding="utf-8")

    converted = convert_xyxy_label(label, image_width=100, image_height=200)

    assert converted == "0 0.200000 0.200000 0.200000 0.200000\n"


def test_detector_training_exposes_reproducible_optional_tuning_controls() -> None:
    args = build_detector_parser().parse_args(
        ["train", "--data", "data.yaml", "--name", "run"]
    )

    assert args.seed == 2026
    assert args.optimizer == "auto"
    assert args.lr0 == pytest.approx(0.01)
    assert args.mosaic == pytest.approx(1.0)


@pytest.mark.parametrize(
    "row, message",
    [
        ("0 nan 0.5 0.2 0.2", "finite"),
        ("0 1.1 0.5 0.2 0.2", "coordinates"),
        ("0 0.5 0.5 0 0.2", "positive"),
        ("0 0.95 0.5 0.2 0.2", "exceeds image width"),
    ],
)
def test_normalized_labels_reject_invalid_boxes(
    tmp_path: Path, row: str, message: str
) -> None:
    label = tmp_path / "label.txt"
    label.write_text(row + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        normalize_yolo_label(label)


def test_practice_dataset_has_fixed_24_page_manifest(tmp_path: Path) -> None:
    source = make_mth1000(tmp_path / "source")
    output = tmp_path / "practice"

    manifest = prepare_mth1000(source, output, mode="practice", materialize="copy")

    assert manifest["size"] == 24
    assert manifest["splits"] == {"all": 24}
    assert len({entry["image"] for entry in manifest["entries"]}) == 24
    data_yaml = (output / "data.yaml").read_text(encoding="utf-8")
    assert "train: images/all" in data_yaml
    assert "val: images/all" in data_yaml
    assert "test: images/all" in data_yaml
    assert all_label_rows_start_with_zero(output / "labels")


def test_full_split_is_deterministic_exhaustive_and_disjoint(tmp_path: Path) -> None:
    source = make_mth1000(tmp_path / "source")
    first = prepare_mth1000(source, tmp_path / "first", mode="full", materialize="copy")
    second = prepare_mth1000(source, tmp_path / "second", mode="full", materialize="copy")

    assert first["splits"] == {"train": 24, "val": 3, "test": 3}
    first_assignments = {(entry["image"], entry["split"]) for entry in first["entries"]}
    second_assignments = {(entry["image"], entry["split"]) for entry in second["entries"]}
    assert first_assignments == second_assignments
    assert len(first_assignments) == 30
    assert all_label_rows_start_with_zero(tmp_path / "first" / "labels")


def test_mth_practice_staging_is_deterministic_and_raw(tmp_path: Path) -> None:
    source = make_mth1000(tmp_path / "mth1200", count=30)

    first = stage_mth_practice_sample(source, tmp_path / "first")
    second = stage_mth_practice_sample(source, tmp_path / "second")

    assert first["entries"] == second["entries"]
    assert len(first["entries"]) == 24
    assert first["source"] == str(source.resolve())
    raw = next((tmp_path / "first/labels").rglob("*.txt"))
    assert raw.read_text(encoding="utf-8").startswith("17 ")


def _organize_practice(stage: Path, destination: Path, manifest: dict[str, object]) -> None:
    splits = ["train"] * 18 + ["val"] * 2 + ["test"] * 4
    for entry, split in zip(manifest["entries"], splits, strict=True):
        relative = Path(entry["image"])
        source_image = stage / "images" / relative
        source_label = stage / "labels" / relative.with_suffix(".txt")
        image_out = destination / "images" / split / relative
        label_out = destination / "labels" / split / relative.with_suffix(".txt")
        image_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, image_out)
        with Image.open(source_image) as image:
            label_out.write_text(
                convert_xyxy_label(
                    source_label, image_width=image.width, image_height=image.height
                ),
                encoding="utf-8",
            )
    (destination / "data.yaml").write_text(
        f"path: {destination.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames:\n  0: character\n",
        encoding="utf-8",
    )


def test_practice_validator_checks_structure_labels_and_manifest(tmp_path: Path) -> None:
    source = make_mth1000(tmp_path / "mth1200", count=30)
    stage = tmp_path / "stage"
    manifest = stage_mth_practice_sample(source, stage)
    dataset = tmp_path / "practice"
    _organize_practice(stage, dataset, manifest)

    result = validate_mth_practice_dataset(dataset, stage / "manifest.json")

    assert result == {
        "valid": True,
        "size": 24,
        "splits": {"train": 18, "val": 2, "test": 4},
    }

    first_label = next((dataset / "labels/train").rglob("*.txt"))
    first_label.write_text(
        first_label.read_text(encoding="utf-8").replace("0 ", "7 ", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="class must be 0"):
        validate_mth_practice_dataset(dataset, stage / "manifest.json")


def test_course_mth_uses_train_source_for_90_10_and_all_test_source(tmp_path: Path) -> None:
    train_source = make_mth1000(tmp_path / "mth1200", count=30)
    test_source = make_mth1000(tmp_path / "mth1000", count=10)

    first = prepare_mth_course_dataset(
        train_source, test_source, tmp_path / "first", materialize="copy"
    )
    second = prepare_mth_course_dataset(
        train_source, test_source, tmp_path / "second", materialize="copy"
    )

    assert first["splits"] == {"train": 27, "val": 3, "test": 10}
    assert first["entries"] == second["entries"]
    assert {entry["collection"] for entry in first["entries"] if entry["split"] == "test"} == {
        "mth1000"
    }
    assert all_label_rows_start_with_zero(tmp_path / "first/labels")


def test_split_pairs_requires_enough_pages(tmp_path: Path) -> None:
    source = make_mth1000(tmp_path / "source", count=9)
    from nom_ocr_lab.detection import discover_detection_pairs

    with pytest.raises(ValueError, match="At least 10"):
        split_pairs(discover_detection_pairs(source))


def test_discovery_rejects_image_without_character_label(tmp_path: Path) -> None:
    source = make_mth1000(tmp_path / "source", count=10)
    (source / "label_char/volume-0/page-000.txt").unlink()
    from nom_ocr_lab.detection import discover_detection_pairs

    with pytest.raises(ValueError, match="without matching label_char labels"):
        discover_detection_pairs(source)


def test_nomnaocr_view_preserves_nested_document_paths(tmp_path: Path) -> None:
    source = make_yolo_source(
        tmp_path / "nomnaocr", ["Book A/page-1", "Book B/volume/page-2"]
    )
    output = tmp_path / "nom-view"

    manifest = prepare_yolo_view(
        source,
        output,
        label_dir="detection_labels",
        materialize="copy",
    )

    assert manifest["splits"] == {"test": 2}
    assert (output / "images/test/Book A/page-1.jpg").is_file()
    assert (output / "labels/test/Book B/volume/page-2.txt").is_file()
    assert all_label_rows_start_with_zero(output / "labels")


def _image_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (40, 60), "white").save(buffer, format="JPEG")
    return buffer.getvalue()


def test_nomnaocr_archive_combines_partitions_without_filename_collisions(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "nomnaocr.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for partition in ("train", "valid"):
            bundle.writestr(f"{partition}/images/page.jpg", _image_bytes())
            bundle.writestr(
                f"{partition}/labels/page.txt",
                "2539 0.5 0.5 0.2 0.3\n",
            )

    manifest = prepare_nomnaocr_archive(archive, tmp_path / "prepared")

    assert manifest["splits"] == {"test": 2}
    assert (tmp_path / "prepared/images/test/train/page.jpg").is_file()
    assert (tmp_path / "prepared/images/test/valid/page.jpg").is_file()
    assert all_label_rows_start_with_zero(tmp_path / "prepared/labels")
    assert "test: images/test" in (tmp_path / "prepared/data.yaml").read_text()


def test_nomnaocr_archive_rejects_missing_labels(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("test/images/page.jpg", _image_bytes())

    with pytest.raises(ValueError, match="no matching label"):
        prepare_nomnaocr_archive(archive, tmp_path / "prepared")


def test_nakagawa_view_uses_supplied_disjoint_splits(tmp_path: Path) -> None:
    source = make_yolo_source(
        tmp_path / "nakagawa", ["page-train", "page-val", "page-test"], label_dir="labels"
    )
    for split in ("train", "val", "test"):
        (source / f"{split}.txt").write_text(
            f"images/page-{split}.jpg\n", encoding="utf-8"
        )

    manifest = prepare_yolo_view(
        source,
        tmp_path / "nakagawa-view",
        label_dir="labels",
        split_files={split: source / f"{split}.txt" for split in ("train", "val", "test")},
        materialize="copy",
    )

    assert manifest["splits"] == {"train": 1, "val": 1, "test": 1}


def test_view_rejects_cross_split_leakage(tmp_path: Path) -> None:
    source = make_yolo_source(tmp_path / "source", ["page"], label_dir="labels")
    train = source / "train.txt"
    val = source / "val.txt"
    train.write_text("images/page.jpg\n", encoding="utf-8")
    val.write_text("images/page.jpg\n", encoding="utf-8")

    with pytest.raises(ValueError, match="occurs in both"):
        prepare_yolo_view(
            source,
            tmp_path / "view",
            label_dir="labels",
            split_files={"train": train, "val": val},
            materialize="copy",
        )


def test_preparation_refuses_shared_and_nonempty_destinations(tmp_path: Path) -> None:
    source = make_mth1000(tmp_path / "source")
    with pytest.raises(ValueError, match="read-only"):
        prepare_mth1000(
            source, Path("/data/shared/project2026/output"), mode="practice"
        )
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_mth1000(source, nonempty, mode="practice")


def test_iou_error_summary_and_gallery(tmp_path: Path) -> None:
    truth = [Box(0.1, 0.1, 0.3, 0.3), Box(0.6, 0.6, 0.8, 0.8)]
    predictions = [Box(0.1, 0.1, 0.3, 0.3), Box(0.0, 0.7, 0.1, 0.8)]
    assert box_iou(truth[0], predictions[0]) == pytest.approx(1.0)
    assert summarize_errors(truth, predictions) == {
        "matched": 1,
        "missed": 1,
        "spurious": 1,
        "poor_localization": 0,
        "merge_candidates": 0,
        "split_candidates": 0,
    }

    dataset = tmp_path / "dataset"
    image = dataset / "images/test/book/page.jpg"
    label = dataset / "labels/test/book/page.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(image)
    label.write_text("0 0.2 0.2 0.2 0.2\n0 0.7 0.7 0.2 0.2\n", encoding="utf-8")
    prediction = tmp_path / "predictions/book/page.txt"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("0 0.2 0.2 0.2 0.2 0.9\n", encoding="utf-8")

    report = create_error_gallery(dataset, tmp_path / "predictions", tmp_path / "gallery")

    assert report["totals"]["missed"] == 1
    assert (tmp_path / "gallery/book/page.jpg").is_file()
    assert json.loads((tmp_path / "gallery/error_summary.json").read_text())["totals"][
        "missed"
    ] == 1

    all_gallery = tmp_path / "all-gallery"
    all_report = create_error_gallery(
        dataset,
        tmp_path / "predictions",
        all_gallery,
        selection="all",
        limit=1,
    )
    assert all_report["selection"] == "all"
    assert len(all_report["gallery_pages"]) == 1


def test_iou_examples_and_one_to_one_matching(tmp_path: Path) -> None:
    identical = Box(0.1, 0.1, 0.3, 0.3)
    disjoint = Box(0.7, 0.7, 0.9, 0.9)
    partial = Box(0.2, 0.2, 0.4, 0.4)
    assert box_iou(identical, identical) == pytest.approx(1.0)
    assert box_iou(identical, disjoint) == 0.0
    assert box_iou(identical, partial) == pytest.approx(1 / 7)

    dataset = tmp_path / "dataset"
    image = dataset / "images/test/book/page.jpg"
    label = dataset / "labels/test/book/page.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(image)
    label.write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    prediction = tmp_path / "predictions/book/page.txt"
    prediction.parent.mkdir(parents=True)
    prediction.write_text(
        "0 0.2 0.2 0.2 0.2 0.9\n0 0.2 0.2 0.2 0.2 0.8\n",
        encoding="utf-8",
    )

    report = evaluate_saved_predictions(
        dataset, tmp_path / "predictions", iou_thresholds=(0.5, 0.75)
    )
    assert (report["tp"], report["fp"], report["fn"]) == (1, 1, 0)
    assert report["precision"] == pytest.approx(0.5)
    assert report["recall"] == pytest.approx(1.0)
    assert report["ap50"] == pytest.approx(1.0)
    assert report["map50_95"] == pytest.approx(1.0)


def test_saved_prediction_metrics_include_empty_nested_pages_and_write_reports(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    for name in ("Book A/page-1", "Book B/part/page-2"):
        image = dataset / "images/test" / f"{name}.jpg"
        label = dataset / "labels/test" / f"{name}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 100), "white").save(image)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    prediction = tmp_path / "predictions/Book A/page-1.txt"
    prediction.parent.mkdir(parents=True)
    prediction.write_text("0 0.5 0.5 0.2 0.2 0.9\n", encoding="utf-8")

    output = tmp_path / "metrics"
    assert evaluate_predictions_main(
        [
            "--dataset",
            str(dataset),
            "--predictions",
            str(tmp_path / "predictions"),
            "--output",
            str(output),
            "--iou-thresholds",
            "0.50,0.75",
        ]
    ) == 0
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert report["pages"] == 2
    assert report["tp"] == 1
    assert report["fn"] == 1
    assert report["recall"] == pytest.approx(0.5)
    assert (output / "metrics.md").read_text(encoding="utf-8").startswith(
        "# Saved-label detection metrics"
    )


def test_confidence_order_controls_ap_and_stricter_iou_reduces_map(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image = dataset / "images/test/page.jpg"
    label = dataset / "labels/test/page.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(image)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    prediction = tmp_path / "predictions/page.txt"
    prediction.parent.mkdir(parents=True)
    prediction.write_text(
        # The false positive is listed second but must be scored first by confidence.
        "0 0.55 0.5 0.2 0.2 0.8\n0 0.1 0.1 0.1 0.1 0.9\n",
        encoding="utf-8",
    )

    report = evaluate_saved_predictions(
        dataset, tmp_path / "predictions", iou_thresholds=(0.5, 0.75)
    )
    assert report["ap50"] == pytest.approx(0.5)
    assert report["per_iou"]["0.75"]["ap"] == 0.0
    assert report["map50_95"] == pytest.approx(0.25)


def test_saved_prediction_validation_and_ap_helpers(tmp_path: Path) -> None:
    assert parse_iou_thresholds("0.50,0.75,0.95") == (0.5, 0.75, 0.95)
    with pytest.raises(ValueError, match="unique and increasing"):
        parse_iou_thresholds("0.75,0.50")
    assert interpolated_average_precision([1.0, 0.5], [0.5, 1.0]) == pytest.approx(
        76 / 101
    )

    label = tmp_path / "prediction.txt"
    invalid_rows = (
        ("0 0.5 0.5 0.2 0.2", "expected 6"),
        ("0 nan 0.5 0.2 0.2 0.9", "finite"),
        ("0 0.5 0.5 0 0.2 0.9", "positive"),
        ("0 0.95 0.5 0.2 0.2 0.9", "exceeds image width"),
        ("0 0.5 0.5 0.2 0.2 1.1", "confidence"),
    )
    for row, message in invalid_rows:
        label.write_text(row + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_yolo_boxes(label, require_confidence=True)

    with pytest.raises(SystemExit, match="read-only"):
        evaluate_predictions_main(
            [
                "--dataset",
                str(tmp_path),
                "--predictions",
                str(tmp_path),
                "--output",
                "/data/shared/project2026/metrics",
            ]
        )


def test_week3_handout_is_explicitly_single_class() -> None:
    handout = (
        Path(__file__).parents[1] / "labs/week-03-detection-1/README.md"
    ).read_text(encoding="utf-8")
    assert "class `0`" in handout
    assert "18 MTH1200 pages" in handout
    assert "2 MTH1200 pages" in handout
    assert "4 MTH1200 pages" in handout
    assert "1,080" in handout and "120 MTH1200" in handout
    assert "canonical" in handout and "MTHv2 research split" in handout
    assert "python -m nom_ocr_lab.prepare_detection_data" in handout
    assert "box_loss" in handout
    assert "last.pt" in handout
    assert "sbatch" in handout
    assert "not a target" in handout
    assert "models/pretrained/yolo11n.pt" in handout
    assert "YOLO(\"yolo11n.pt\")" in handout
    assert "yolo11n.pt is ready" in handout


def test_week4_handout_uses_week3_checkpoint_and_shared_source_tools() -> None:
    handout = (
        Path(__file__).parents[1] / "labs/week-04-detection-2/README.md"
    ).read_text(encoding="utf-8")
    assert "week03-mth1200/weights/best.pt" in handout
    assert "prepare-nakagawa" in handout
    assert "prepare-nomnaocr-archive" in handout
    assert "Reference only — not a reproduction target" in handout
    assert "0.9875" in handout and "0.9674" in handout
    assert "sbatch" in handout
    assert "Do not expect your values to match" in handout
