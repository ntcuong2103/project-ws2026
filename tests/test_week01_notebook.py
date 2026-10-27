from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from PIL import Image


@pytest.mark.parametrize(
    ("job_id", "expected_status"),
    [(None, "outside a Slurm allocation"), ("1234", "Slurm allocation: job 1234")],
)
def test_week01_notebook_reports_allocation_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, job_id: str | None, expected_status: str
) -> None:
    sample_root = tmp_path / "lab-exercise" / "datasets" / "nomnaocr-sample"
    for index in range(5):
        image_path = sample_root / "images" / f"page-{index}.jpg"
        label_path = sample_root / "line_labels" / f"page-{index}.txt"
        detection_label_path = sample_root / "detection_labels" / f"page-{index}.txt"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        detection_label_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 30), "white").save(image_path)
        label_path.write_text("0,0,10,0,10,20,0,20,Nôm\n", encoding="utf-8")
        detection_label_path.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    monkeypatch.setenv("LAB_WORKSPACE", str(tmp_path / "lab-exercise"))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    if job_id is None:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    else:
        monkeypatch.setenv("SLURM_JOB_ID", job_id)
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    notebook_path = (
        Path(__file__).parents[1]
        / "labs"
        / "week-01-orientation"
        / "visualize_sample_page.ipynb"
    )
    notebook = nbformat.read(notebook_path, as_version=4)

    executed = NotebookClient(notebook, timeout=120, kernel_name="python3").execute(
        cwd=str(notebook_path.parents[2])
    )
    output_text = "".join(
        output.get("text", "")
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.output_type == "stream"
    )
    assert expected_status in output_text
    assert "matched page triplets" in output_text
    assert "character-position boxes" in output_text
