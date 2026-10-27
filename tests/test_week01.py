from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from nom_ocr_lab.annotations import (
    parse_detection_annotation,
    parse_line_annotation,
    read_detection_annotations,
    read_line_annotations,
)
from nom_ocr_lab.prepare_sample import main as prepare_sample_main
from nom_ocr_lab.sample import discover_page_pairs, prepare_sample


LABEL = "0,0,10,0,10,20,0,20,Nôm"
DETECTION_LABEL = "0 0.5 0.5 0.5 0.5"


def make_source(root: Path, names: list[str]) -> Path:
    for name in names:
        image_path = root / "images" / f"{name}.jpg"
        label_path = root / "line_labels" / f"{name}.txt"
        detection_label_path = root / "detection_labels" / f"{name}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        detection_label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 24), "white").save(image_path)
        label_path.write_text(LABEL + "\n", encoding="utf-8")
        detection_label_path.write_text(DETECTION_LABEL + "\n", encoding="utf-8")
    return root


def test_discover_page_pairs_is_deterministic(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["z/page", "a/page", "middle"])

    pairs = discover_page_pairs(source)

    assert [pair.relative_image.as_posix() for pair in pairs] == [
        "a/page.jpg",
        "middle.jpg",
        "z/page.jpg",
    ]


def test_discover_page_pairs_requires_complete_triplets(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["complete", "missing-detection", "missing-line"])
    (source / "detection_labels" / "missing-detection.txt").unlink()
    (source / "line_labels" / "missing-line.txt").unlink()

    pairs = discover_page_pairs(source)

    assert [pair.relative_image.as_posix() for pair in pairs] == ["complete.jpg"]


def test_prepare_sample_creates_manifest_and_reuses_valid_sample(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", [f"page-{index}" for index in range(6)])
    destination = tmp_path / "workspace" / "datasets" / "nomnaocr-sample"

    first_pairs, first_created = prepare_sample(source, destination, size=5)
    second_pairs, second_created = prepare_sample(source, destination, size=5)

    assert first_created is True
    assert second_created is False
    assert [pair.relative_image for pair in first_pairs] == [
        pair.relative_image for pair in second_pairs
    ]
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == 2
    assert manifest["size"] == 5
    assert len(manifest["pairs"]) == 5
    assert set(manifest["pairs"][0]) == {
        "image",
        "image_sha256",
        "line_label",
        "line_label_sha256",
        "detection_label",
        "detection_label_sha256",
    }
    assert first_pairs[0].detection_label_path.is_file()


def test_prepare_sample_rejects_insufficient_complete_triplets(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["complete", "incomplete"])
    (source / "detection_labels" / "incomplete.txt").unlink()

    with pytest.raises(RuntimeError, match="only 1 complete page triplet"):
        prepare_sample(source, tmp_path / "destination", size=2)


def test_prepare_sample_rejects_partial_destination(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["page"])
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "unexpected.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_sample(source, destination, size=1)


def test_prepare_sample_rejects_version_one_manifest(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["page"])
    destination = tmp_path / "destination"
    prepare_sample(source, destination, size=1)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported sample manifest version"):
        prepare_sample(source, destination, size=1)


def test_prepare_sample_rejects_destination_inside_source(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["page"])

    with pytest.raises(ValueError, match="must not be inside"):
        prepare_sample(source, source / "derived", size=1)


def test_prepare_sample_rejects_read_only_shared_destination(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["page"])

    with pytest.raises(ValueError, match="read-only"):
        prepare_sample(source, Path("/data/shared/project2026/lab-output"), size=1)


def test_prepare_sample_detects_changed_file(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["page"])
    destination = tmp_path / "destination"
    prepare_sample(source, destination, size=1)
    pair = discover_page_pairs(destination)[0]
    pair.label_path.write_text("changed", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its manifest"):
        prepare_sample(source, destination, size=1)


def test_prepare_sample_detects_changed_detection_label(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["page"])
    destination = tmp_path / "destination"
    prepare_sample(source, destination, size=1)
    pair = discover_page_pairs(destination)[0]
    pair.detection_label_path.write_text("0 0.4 0.4 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from its manifest"):
        prepare_sample(source, destination, size=1)


def test_prepare_sample_detects_missing_detection_label(tmp_path: Path) -> None:
    source = make_source(tmp_path / "source", ["page"])
    destination = tmp_path / "destination"
    prepare_sample(source, destination, size=1)
    (destination / "detection_labels" / "page.txt").unlink()

    with pytest.raises(ValueError, match="recorded file is missing"):
        prepare_sample(source, destination, size=1)


def test_parse_and_read_line_annotations(tmp_path: Path) -> None:
    annotation = parse_line_annotation(LABEL)
    assert annotation.points == ((0.0, 0.0), (10.0, 0.0), (10.0, 20.0), (0.0, 20.0))
    assert annotation.text == "Nôm"

    label_path = tmp_path / "page.txt"
    label_path.write_text("\n" + LABEL + "\n", encoding="utf-8")
    assert read_line_annotations(label_path) == [annotation]


def test_parse_and_read_detection_annotations(tmp_path: Path) -> None:
    annotation = parse_detection_annotation(DETECTION_LABEL)
    assert annotation.class_id == 0
    assert annotation.x_center == 0.5
    assert annotation.height == 0.5

    label_path = tmp_path / "page.txt"
    label_path.write_text("\n" + DETECTION_LABEL + "\n", encoding="utf-8")
    assert read_detection_annotations(label_path) == [annotation]


@pytest.mark.parametrize(
    "raw_line",
    ["0,0,10,text", "x,0,10,0,10,20,0,20,text"],
)
def test_malformed_annotations_fail_clearly(raw_line: str) -> None:
    with pytest.raises(ValueError):
        parse_line_annotation(raw_line)


def test_cli_uses_canonical_workspace_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = make_source(tmp_path / "source", ["page"])
    workspace = tmp_path / "lab-exercise"

    result = prepare_sample_main(
        ["--source", str(source), "--workspace", str(workspace), "--size", "1"]
    )

    assert result == 0
    assert (workspace / "datasets" / "nomnaocr-sample" / "manifest.json").is_file()
    assert "Created 1 page triplets" in capsys.readouterr().out


def test_cli_accepts_workspace_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = make_source(tmp_path / "source", ["page"])
    workspace = tmp_path / "lab-exercise"
    monkeypatch.setenv("LAB_WORKSPACE", str(workspace))

    prepare_sample_main(["--source", str(source), "--size", "1"])

    assert (workspace / "datasets" / "nomnaocr-sample" / "manifest.json").is_file()


def test_week01_batch_job_is_cpu_only() -> None:
    script = (
        Path(__file__).parents[1]
        / "labs"
        / "week-01-orientation"
        / "prepare_sample.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --cpus-per-task=2" in script
    assert "#SBATCH --mem=4G" in script
    assert "#SBATCH --time=00:10:00" in script
    assert "#SBATCH --gres" not in script
    assert "uv run python labs/week-01-orientation/prepare_sample.py" in script
