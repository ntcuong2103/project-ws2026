# Week 5 — Recognition I: IDS and Nôm Preparation

[← Back to course README](../../README.md) · Previous: [Week 4](../week-04-detection-2/) · Next: [Week 6](../week-06-recognition-2/)

> **Interactive Slurm reminder:** Whenever you use `srun --pty ...`, run `exit` as soon as you finish. This releases the allocation; do not leave an interactive session idle.

**Session:** each group reviews Week 4 (5 min/group), then completes this lab.

## Goals

By the end of the lab, you can:

- explain how an Ideographic Description Sequence (IDS) represents character structure;
- distinguish IDS operators from terminal components;
- convert Unicode Chinese and Nôm text to the recognizer's expanded IDS tokens;
- measure dictionary coverage on one real NomNaOCR page; and
- export one page of detector boxes as unlabeled crops for recognition.

This is **text-level IDS tokenization**, not visual radical segmentation. The supplied
repositories do not contain radical bounding boxes or a model that cuts a character image
into radical regions. The recognizer learns to predict IDS tokens from the complete crop.

## Required reading

1. Tomohiko Morioka, [“CHISE: Character Processing Based on Character Ontology”](https://link.springer.com/chapter/10.1007/978-3-540-78159-2_14), LKR 2008, pp. 148–162.
2. Jianshu Zhang, Yixing Zhu, Jun Du, and Lirong Dai, [“Radical analysis network for zero-shot learning in printed Chinese character recognition”](https://arxiv.org/abs/1711.01889), ICME 2018.

Read for two ideas: why coded character sets alone are insufficient for historical character
work, and why predicting a shared structural vocabulary can recognize more classes than an
atomic one-class-per-character formulation.

## Data references and source of truth

- [cjkvi/cjkvi-ids](https://github.com/cjkvi/cjkvi-ids): direct/raw IDS data, including
  `ids.txt` and the requested `ids-ext-cdef.txt`.
- [chise/ids](https://github.com/chise/ids): the CHISE IDS database and ontology source.
- [yi-bai/ids](https://github.com/yi-bai/ids): alternative IDS lists at several normalization
  levels.
- [zi.tools](https://zi.tools/): interactive Unicode, glyph, and IDS lookup for manual checks.

For executable course results, do not merge or reorder these resources yourself. The source
of truth is `ntcuong2103/nom-ids` release commit
`6f7f7f8242e95a95eb6920092cd290d9e4e49deb`:

- `ids_exp.txt` maps one Unicode character to its recursively expanded IDS;
- `vocab_ids.txt` gives the exact checkpoint-sensitive token order.

Reverse lookup must map one IDS to a list of candidate characters: distinct Unicode
characters can share the same expanded sequence, so a plain `{IDS: character}` dictionary
would silently discard candidates.

The crop convention comes from `ntcuong2103/nom-ocr-inference` commit
`24f417ec8b58590b7786a186915badcee625df00`. CJKVI commit
`86b4d16159f0079437870408f0ca186e529015db` is downloaded for provenance and comparison.

## 1. What is an IDS?

An IDS is a prefix expression. An Ideographic Description Character (IDC) states a spatial
relationship and is followed by its operands:

- binary examples: `⿰` left–right, `⿱` above–below, `⿴` surround, `⿻` overlap;
- ternary examples: `⿲` left–middle–right and `⿳` above–middle–below.

For example, `明` has direct IDS `⿰日月`: place `日` left of `月`. Nested structures remain
prefix expressions. “Radical vocabulary” is useful shorthand, but the model vocabulary also
contains IDCs, strokes, components, and terminal ideographs that are not radicals in the
strict linguistic sense.

There are two relevant representations:

- **Direct/raw IDS:** one decomposition layer from CJKVI; components may themselves be
  decomposable.
- **Expanded IDS:** recursively replaces decomposable components using the convention used
  to train the recognizer. This is the ground-truth sequence used by the code.

## 2. Download the pinned resources

Start from the course repository in a shell where `LAB_WORKSPACE` is set:

```bash
uv run python -m nom_ocr_lab.prepare_recognition_resources \
  --output "$LAB_WORKSPACE/datasets/ids"
```

The command downloads all four files, verifies their SHA-256 checksums, and records their
URLs, commits, sizes, and hashes in `manifest.json`. A matching file is reused. A mismatched
file is never silently replaced. Generated data belongs under `$LAB_WORKSPACE`, not in Git.

## 3. Run the guided notebook

Open [ids_and_nomnaocr_crops.ipynb](ids_and_nomnaocr_crops.ipynb). It provides a runnable
reference implementation and walks through:

1. direct versus expanded IDS;
2. the exact expected conversions for `休好字明林`;
3. Unicode-code-point splitting of a line transcription;
4. coverage and OOV reporting for one Week 1 sample page;
5. manual verification of at least five covered Nôm characters with zi.tools; and
6. a contact sheet of the detector crops exported below.

Reimplement the demonstrated text-to-IDS converter in your team's separate project
repository. Compare its structured output with the reference notebook. Preserve character
boundaries; do not concatenate a whole line without retaining which IDS belongs to which
Unicode code point.

zi.tools may show a direct IDS, a glyph-region variant, or another normalization. Record such
differences rather than changing the authoritative expanded course dictionary to force a
match. zi.tools is a manual reference, not an automated dependency.

## 4. Analyze one real NomNaOCR page

Choose one of the five durable Week 1 pages below
`$LAB_WORKSPACE/datasets/nomnaocr-sample`. Give the notebook its relative image path. It will
find the matching `line_labels/<relative-path>.txt`, parse all transcriptions, and report each
non-whitespace code point as covered or OOV.

A line label contains a polygon and a line transcription. It does **not** say which detected
box owns which character. Do not assign transcription characters directly to detector boxes.
Missed, duplicate, split, or merged detections make positional pairing unreliable; sequence
alignment handles that in Weeks 9–11.

## 5. Export one page of unlabeled inference crops

Run the stronger Week 3 or Week 4 detector on the selected page. Use a new run name so the
detector's non-overwrite rule protects earlier predictions:

```bash
uv run python -m nom_ocr_lab.run_detector predict \
  --source "$LAB_WORKSPACE/datasets/nomnaocr-sample/images/<relative-page>.jpg" \
  --model "$LAB_WORKSPACE/models/week04-nakagawa/weights/best.pt" \
  --name week05-one-page --imgsz 1280 --batch 1 --device 0
```

Then pass the same image and its generated prediction label to the supplied exporter:

```bash
uv run python -m nom_ocr_lab.export_recognition_crops \
  --image "$LAB_WORKSPACE/datasets/nomnaocr-sample/images/<relative-page>.jpg" \
  --labels "$LAB_WORKSPACE/predictions/week05-one-page/labels/<page-name>.txt" \
  --output "$LAB_WORKSPACE/datasets/nomnaocr-recognition-crops/<page-name>" \
  --expand-ratio 1.2
```

The exporter accepts ground-truth-style five-field YOLO rows and prediction rows with a sixth
confidence field. It retains `box_id`, expands each box to a clipped square by `1.2`, writes
RGB PNG crops, and creates both JSON and TSV manifests. These crops deliberately have no
character identity. Week 6 applies the recognizer's tensor preprocessing.

Use an interactive CPU allocation for the notebook and submit detector inference through the
same Slurm policy used in Weeks 3–4. Do not request a GPU for dictionary conversion or crop
visualization.

## Tasks

- [ ] Read both assigned papers and explain direct versus expanded IDS.
- [ ] Download and verify the pinned IDS resources.
- [ ] Reproduce the five exact Chinese reference conversions.
- [ ] Reimplement text→IDS conversion in the team project repository.
- [ ] Produce a one-page Nôm coverage/OOV table.
- [ ] Record five zi.tools checks, including any normalization or glyph-variant difference.
- [ ] Run the Week 3–4 detector on that page and inspect the exported crop contact sheet.
- [ ] Explain why the crops remain unlabeled until sequence alignment.

## Deliverable

A validated IDS converter, one-page transcription coverage report, five-row zi.tools
verification table, and unlabeled recognizer inference crops with their manifest and contact
sheet. Do not commit downloaded dictionaries, predictions, or crops.
