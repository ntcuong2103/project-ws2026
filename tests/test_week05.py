from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from PIL import Image
import pytest

import nom_ocr_lab.recognition as recognition
from nom_ocr_lab.recognition import (
    ResourceSpec,
    load_cjkvi_ids,
    load_expanded_ids,
    load_vocabulary,
    prepare_resources,
    reverse_ids,
    text_to_ids,
)
from nom_ocr_lab.recognition_crops import (
    expanded_crop_box,
    export_crops,
    parse_prediction,
)


class Download(BytesIO):
    def __enter__(self) -> "Download":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_prepare_resources_verifies_reuses_and_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"character\tids\n"
    checksum = recognition.hashlib.sha256(payload).hexdigest()
    spec = ResourceSpec("example/repo", "abc123", "nom-ids/ids_exp.txt", "mock://ids", checksum)
    monkeypatch.setattr(recognition, "RESOURCE_SPECS", (spec,))
    calls = []

    def opener(url: str) -> Download:
        calls.append(url)
        return Download(payload)

    first = prepare_resources(tmp_path / "ids", opener=opener)
    second = prepare_resources(tmp_path / "ids", opener=opener)

    assert calls == ["mock://ids"]
    assert first["resources"][0]["reused"] is False
    assert second["resources"][0]["reused"] is True
    stored = json.loads((tmp_path / "ids" / "manifest.json").read_text())
    assert stored["resources"][0]["commit"] == "abc123"
    assert stored["resources"][0]["sha256"] == checksum


def test_prepare_resources_refuses_bad_download_and_existing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = ResourceSpec("example/repo", "abc123", "file.txt", "mock://ids", "0" * 64)
    monkeypatch.setattr(recognition, "RESOURCE_SPECS", (spec,))
    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare_resources(tmp_path / "download", opener=lambda _: Download(b"wrong"))
    assert not (tmp_path / "download" / "file.txt.part").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "file.txt").write_bytes(b"wrong")
    with pytest.raises(FileExistsError, match="checksum-mismatched"):
        prepare_resources(existing, opener=lambda _: Download(b"unused"))


def test_dictionary_loaders_conversion_and_reverse_mapping(tmp_path: Path) -> None:
    expanded_path = tmp_path / "ids_exp.txt"
    expanded_path.write_text(
        "休\t⿰亻木\n好\t⿰女子\n字\t⿱宀子\n明\t⿰日月\n林\t⿰木木\n朋\t⿰月月\n倗\t⿰月月\n",
        encoding="utf-8",
    )
    vocab_path = tmp_path / "vocab_ids.txt"
    vocab_path.write_text("木\n⿰\n亻\n月\n", encoding="utf-8")
    cjkvi_path = tmp_path / "ids.txt"
    cjkvi_path.write_text(
        "# comment\nU+660E\t明\t⿰日月[TV]\t⿱日月[J]\n", encoding="utf-8"
    )

    mapping = load_expanded_ids(expanded_path)
    assert {character: mapping[character] for character in "休好字明林"} == {
        "休": "⿰亻木",
        "好": "⿰女子",
        "字": "⿱宀子",
        "明": "⿰日月",
        "林": "⿰木木",
    }
    assert load_vocabulary(vocab_path) == ["木", "⿰", "亻", "月"]
    assert load_cjkvi_ids(cjkvi_path) == {"明": "⿰日月"}
    assert reverse_ids(mapping)["⿰月月"] == ("朋", "倗")

    records = text_to_ids("明 X", mapping)
    assert [record.character for record in records] == ["明", "X"]
    assert records[0].codepoint == "U+660E"
    assert records[0].tokens == ("⿰", "日", "月")
    assert records[0].status == "covered"
    assert records[1].ids is None
    assert records[1].status == "oov"


@pytest.mark.parametrize(
    ("row", "confidence"),
    [
        ("0 0.5 0.5 0.2 0.4", None),
        ("0 0.5 0.5 0.2 0.4 0.875", 0.875),
    ],
)
def test_prediction_parser_accepts_five_and_six_fields(
    row: str, confidence: float | None
) -> None:
    assert parse_prediction(row).confidence == confidence


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("0 0.5 0.5 0.2", "5 or 6"),
        ("0 nan 0.5 0.2 0.2", "finite"),
        ("0 0.1 0.5 0.4 0.2", "outside image width"),
        ("0 0.5 0.5 0.2 0.2 1.1", "confidence"),
    ],
)
def test_prediction_parser_rejects_invalid_rows(row: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_prediction(row)


def test_crop_export_preserves_ids_confidence_and_clips(tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (100, 80), "white").save(image_path)
    labels_path = tmp_path / "page.txt"
    labels_path.write_text(
        "0 0.20 0.25 0.20 0.25 0.90\n0 0.75 0.50 0.10 0.25\n",
        encoding="utf-8",
    )

    manifest = export_crops(image_path, labels_path, tmp_path / "crops", expand_ratio=1.2)

    assert manifest["kind"] == "unlabeled-recognizer-inference-crops"
    assert manifest["labeled"] is False
    assert [entry["box_id"] for entry in manifest["entries"]] == [0, 1]
    assert manifest["entries"][0]["confidence"] == 0.9
    assert manifest["entries"][1]["confidence"] is None
    assert (tmp_path / "crops" / "crops" / "box-0000.png").is_file()
    assert (tmp_path / "crops" / "manifest.tsv").is_file()
    with pytest.raises(FileExistsError, match="non-empty"):
        export_crops(image_path, labels_path, tmp_path / "crops")

    edge = parse_prediction("0 0.1 0.1 0.2 0.2")
    _, expanded = expanded_crop_box(
        edge, image_width=100, image_height=80, expand_ratio=1.2
    )
    assert expanded[0] == 0 or expanded[1] == 0


def test_crop_export_validates_before_creating_output(tmp_path: Path) -> None:
    image_path = tmp_path / "page.jpg"
    Image.new("RGB", (10, 10), "white").save(image_path)
    labels_path = tmp_path / "bad.txt"
    labels_path.write_text("0 0.5 0.5 0.01 0.01\n", encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="empty pixel box"):
        export_crops(image_path, labels_path, output)
    assert not output.exists()


def test_week05_handout_contains_required_sources_and_safety_boundary() -> None:
    handout = (
        Path(__file__).parents[1] / "labs" / "week-05-recognition-1" / "README.md"
    ).read_text(encoding="utf-8")
    for expected in (
        "CHISE: Character Processing Based on Character Ontology",
        "Radical analysis network for zero-shot learning",
        "cjkvi/cjkvi-ids",
        "chise/ids",
        "yi-bai/ids",
        "zi.tools",
        "ids_exp.txt",
        "vocab_ids.txt",
        "Do not assign",
    ):
        assert expected in handout


def test_week05_notebook_executes_on_small_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "lab-exercise"
    ids_root = workspace / "datasets" / "ids"
    (ids_root / "nom-ids").mkdir(parents=True)
    (ids_root / "cjkvi").mkdir(parents=True)
    expanded_rows = {
        "休": "⿰亻木",
        "好": "⿰女子",
        "字": "⿱宀子",
        "明": "⿰日月",
        "林": "⿰木木",
    }
    (ids_root / "nom-ids" / "ids_exp.txt").write_text(
        "".join(f"{character}\t{ids}\n" for character, ids in expanded_rows.items()),
        encoding="utf-8",
    )
    (ids_root / "nom-ids" / "vocab_ids.txt").write_text(
        "⿰\n⿱\n亻\n木\n女\n子\n宀\n日\n月\n", encoding="utf-8"
    )
    (ids_root / "cjkvi" / "ids.txt").write_text(
        "".join(
            f"U+{ord(character):04X}\t{character}\t{ids}\n"
            for character, ids in expanded_rows.items()
        ),
        encoding="utf-8",
    )
    (ids_root / "cjkvi" / "ids-ext-cdef.txt").write_text("", encoding="utf-8")

    sample_root = workspace / "datasets" / "nomnaocr-sample"
    image_path = sample_root / "images" / "volume" / "page.jpg"
    line_path = sample_root / "line_labels" / "volume" / "page.txt"
    image_path.parent.mkdir(parents=True)
    line_path.parent.mkdir(parents=True)
    Image.new("RGB", (40, 60), "white").save(image_path)
    line_path.write_text("0,0,20,0,20,50,0,50,明林\n", encoding="utf-8")

    prediction_path = tmp_path / "prediction.txt"
    prediction_path.write_text("0 0.5 0.5 0.2 0.3 0.9\n", encoding="utf-8")
    export_crops(
        image_path,
        prediction_path,
        workspace / "datasets" / "nomnaocr-recognition-crops" / "page",
    )

    monkeypatch.setenv("LAB_WORKSPACE", str(workspace))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    notebook_path = (
        Path(__file__).parents[1]
        / "labs"
        / "week-05-recognition-1"
        / "ids_and_nomnaocr_crops.ipynb"
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
    assert "Five Chinese reference conversions passed." in output_text
    assert "Page lookup coverage: 2/2 = 100.00%" in output_text
    assert "Displayed 1 of 1 crops." in output_text
