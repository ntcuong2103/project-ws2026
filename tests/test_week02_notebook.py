from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from PIL import Image


def _make_dataset(root: Path) -> None:
    for index in range(5):
        relative_image = Path("volume") / f"page-{index}.jpg"
        image_path = root / "images" / relative_image
        line_path = root / "line_labels" / relative_image.with_suffix(".txt")
        detection_path = root / "detection_labels" / relative_image.with_suffix(".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        line_path.parent.mkdir(parents=True, exist_ok=True)
        detection_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (40, 60), "white").save(image_path)
        line_path.write_text(
            f"0,0,20,0,20,50,0,50,Nôm{index}\n", encoding="utf-8"
        )
        detection_path.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")


def test_week02_notebook_executes_on_five_page_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_root = tmp_path / "full"
    sample_root = tmp_path / "lab-exercise" / "datasets" / "nomnaocr-sample"
    _make_dataset(full_root)
    _make_dataset(sample_root)
    (sample_root / "manifest.json").write_text(
        json.dumps({"size": 5}), encoding="utf-8"
    )

    monkeypatch.setenv("LAB_WORKSPACE", str(tmp_path / "lab-exercise"))
    monkeypatch.setenv("NOMNAOCR_ROOT", str(full_root))
    monkeypatch.setenv("MPLBACKEND", "Agg")

    notebook_path = (
        Path(__file__).parents[1]
        / "labs"
        / "week-02-foundations"
        / "nomnaocr_eda_and_edit_distance.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)
    executed = NotebookClient(
        notebook, timeout=120, kernel_name="python3", allow_errors=False
    ).execute(cwd=str(notebook_path.parents[2]))

    output_text = "".join(
        output.get("text", "")
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert "Matched sample triplets: 5" in output_text
    assert "All distance sanity checks passed." in output_text
    assert "Normalized Levenshtein" in output_text


def test_week02_handout_includes_readings_and_sample_recovery_command() -> None:
    handout = (
        Path(__file__).parents[1]
        / "labs"
        / "week-02-foundations"
        / "README.md"
    ).read_text(encoding="utf-8")

    assert "NomNaOCR: The First Dataset" in handout
    assert "Automated Character-Level Annotation" in handout
    assert "prepare_sample.sbatch" in handout
    assert "character-position detection" in handout
    assert "pseudo ground truth" in handout
    assert "recognition/alignment" in handout
