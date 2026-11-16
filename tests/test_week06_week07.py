from __future__ import annotations

import json
import math
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from PIL import Image
import pytest

from nom_ocr_lab.recognition_decoding import (
    bidirectional_greedy_decode,
    deduplicate_sequences,
    greedy_decode,
    greedy_decode_direction,
    ids_candidates,
    sequence_log_prob,
)
from nom_ocr_lab.recognition_runtime import RecognizerRuntime, load_config
from nom_ocr_lab.run_detector import _prediction_manifest


def make_config(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "images" / "book").mkdir(parents=True)
    (root / "labels" / "book").mkdir(parents=True)
    (root / "ids").mkdir()
    Image.new("RGB", (30, 20), "white").save(root / "images" / "book" / "page.jpg")
    (root / "labels" / "book" / "page.txt").write_text("0 0.5 0.5 0.4 0.5 0.8\n")
    (root / "ids" / "vocab.txt").write_text("<pad>\n<sos>\n<eos>\n木\n", encoding="utf-8")
    (root / "ids" / "expanded.txt").write_text("木\t木\n", encoding="utf-8")
    (root / "detector.pt").write_text("x")
    (root / "recognizer.ckpt").write_text("x")
    config = tmp_path / "recognition.yaml"
    config.write_text(
        """paths:
  detector_checkpoint: ${LAB_WORKSPACE}/detector.pt
  recognizer_checkpoint: ${LAB_WORKSPACE}/recognizer.ckpt
  vocabulary: ${LAB_WORKSPACE}/ids/vocab.txt
  expanded_ids: ${LAB_WORKSPACE}/ids/expanded.txt
  images: ${LAB_WORKSPACE}/images
  detector_labels: ${LAB_WORKSPACE}/labels
  outputs: ${LAB_WORKSPACE}/outputs
model:
  class_path: fake:Backend
  sos_idx: 1
  eos_idx: 2
  pad_idx: 0
  vocab_size: 7
preprocessing:
  image_size: 16
  expand_ratio: 1.2
  invert: true
inference:
  max_length: 5
""",
        encoding="utf-8",
    )
    return config


def test_yaml_expansion_crop_metadata_and_fake_logits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_WORKSPACE", str(tmp_path / "workspace"))
    config = load_config(make_config(tmp_path))
    class Backend:
        def logits(self, crops: object, prefixes: object) -> object:
            import torch
            return torch.zeros((1, 2, 4))

    runtime = RecognizerRuntime(config, backend=Backend())
    assert tuple(runtime.teacher_forced_logits(None, None).shape) == (1, 2, 4)
    assert tuple(runtime.next_token_logits(None, None).shape) == (1, 4)


def test_config_rejects_vocab_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_WORKSPACE", str(tmp_path / "workspace"))
    config = make_config(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace("vocab_size: 7", "vocab_size: 3"), encoding="utf-8")
    with pytest.raises(ValueError, match="vocab_size"):
        load_config(config)


def test_decoder_fixtures_exactly_match() -> None:
    path = Path(__file__).parents[1] / "labs/week-07-recognition-3/greedy_decode_fixtures.json"
    for fixture in json.loads(path.read_text(encoding="utf-8")):
        rows = iter(fixture["logits"])
        actual = greedy_decode(lambda _prefix: next(rows), sos_idx=fixture["sos_idx"], eos_idx=fixture["eos_idx"], max_length=fixture["max_length"])
        assert actual == fixture["expected"], fixture["name"]


def test_l2r_direction_matches_plain_greedy_fixtures() -> None:
    path = Path(__file__).parents[1] / "labs/week-07-recognition-3/greedy_decode_fixtures.json"
    for fixture in json.loads(path.read_text(encoding="utf-8")):
        rows = iter(fixture["logits"])
        tokens, _own = greedy_decode_direction(
            lambda _prefix: next(rows), direction="l2r", sos_idx=fixture["sos_idx"],
            eos_idx=fixture["eos_idx"], max_length=fixture["max_length"],
        )
        assert list(tokens) == fixture["expected"], fixture["name"]


def _fixture_logits(fixture: dict) -> object:
    table = fixture["next_probabilities"]
    return lambda prefix: [math.log(p) for p in table[",".join(map(str, prefix))]]


def test_bidirectional_fixtures_exactly_match() -> None:
    path = Path(__file__).parents[1] / "labs/week-07-recognition-3/bidirectional_decode_fixtures.json"
    fixtures = json.loads(path.read_text(encoding="utf-8"))
    assert {f["name"] for f in fixtures} == {"r2l_wins", "l2r_wins", "agree_tie_keeps_l2r", "r2l_max_length"}
    for fixture in fixtures:
        result = bidirectional_greedy_decode(
            _fixture_logits(fixture), sos_idx=fixture["sos_idx"], eos_idx=fixture["eos_idx"],
            max_length=fixture["max_length"], alpha=fixture["alpha"],
        )
        expected = fixture["expected"]
        assert result["best"] == expected["best"], fixture["name"]
        assert list(result["tokens"]) == expected["tokens"], fixture["name"]
        for direction in ("l2r", "r2l"):
            actual = result[direction]
            assert list(actual.tokens) == expected[direction]["tokens"], (fixture["name"], direction)
            for field in ("own", "cross", "score"):
                assert getattr(actual, field) == pytest.approx(expected[direction][field], abs=1e-3), (fixture["name"], direction, field)


def test_r2l_rescoring_reads_tokens_backwards() -> None:
    fixture = json.loads((Path(__file__).parents[1] / "labs/week-07-recognition-3/bidirectional_decode_fixtures.json").read_text(encoding="utf-8"))[0]
    logits = _fixture_logits(fixture)
    # P_r2l(⿰木人) = P(人|EOS) P(木|EOS 人) P(⿰|EOS 人 木) P(SOS|EOS 人 木 ⿰) = .8 * .8 * .9 * .9
    assert sequence_log_prob(logits, [3, 4, 5], direction="r2l", sos_idx=1, eos_idx=2) == pytest.approx(math.log(0.8 * 0.8 * 0.9 * 0.9))


def test_deduplication_and_ambiguous_ids() -> None:
    unique, mapping = deduplicate_sequences([[3, 4], [3, 4], [5]])
    assert unique == [(3, 4), (5,)]
    assert mapping == [0, 0, 1]
    assert ids_candidates("⿰月月", {"⿰月月": ("朋", "倗")}) == ("朋", "倗")
    assert ids_candidates("missing", {}) == ()


def test_prediction_manifest_preserves_nested_no_detection_pages(tmp_path: Path) -> None:
    class Args:
        imgsz = 1280
        batch = 8
        device = "0"
        conf = 0.25
        iou = 0.5
        max_det = 1000

    manifest = _prediction_manifest(source_root=tmp_path / "images", model=tmp_path / "model.pt", args=Args(), images=[Path("Book A/page.jpg"), Path("Book B/volume/empty.png")])
    assert manifest["pages"] == [
        {"page_id": "Book A/page", "image": "Book A/page.jpg", "labels": "Book A/page.txt"},
        {"page_id": "Book B/volume/empty", "image": "Book B/volume/empty.png", "labels": "Book B/volume/empty.txt"},
    ]


def test_week_handouts_and_midterm_rules_are_linked() -> None:
    root = Path(__file__).parents[1]
    week6 = (root / "labs/week-06-recognition-2/README.md").read_text(encoding="utf-8")
    week7 = (root / "labs/week-07-recognition-3/README.md").read_text(encoding="utf-8")
    midterm = (root / "docs/MIDTERM-PRESENTATION.md").read_text(encoding="utf-8")
    assert "[batch, step, vocabulary]" in week6
    assert "immediate EOS" in week7
    assert "bidirectional" in week7
    assert "evaluate.ipynb" in week7
    assert "15-minute" in midterm
    assert "Each listed member" in midterm


def test_week06_logits_notebook_executes() -> None:
    root = Path(__file__).parents[1]
    notebook_path = root / "labs/week-06-recognition-2/logits_tutorial.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    executed = NotebookClient(notebook, timeout=120, kernel_name="python3", allow_errors=False).execute(cwd=str(root))
    output = "".join(
        item.get("text", "")
        for cell in executed.cells
        for item in cell.get("outputs", [])
        if item.output_type == "stream"
    )
    assert "Tiny logits inspection passed." in output
