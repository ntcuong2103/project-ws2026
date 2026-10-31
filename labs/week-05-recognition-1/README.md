# Week 5 — Recognition I: IDS and Nôm Preparation

[← Back to course README](../../README.md) · Previous: [Week 4](../week-04-detection-2/) · Next: [Week 6](../week-06-recognition-2/)

> **Interactive Slurm reminder:** Whenever you use `srun --pty ...`, run `exit` as soon as you finish. This releases the allocation; do not leave an interactive session idle.

**Session:** each group begins with a five-minute Week 4 review, then works through this lab.

This week moves from finding characters on a page to describing what a character is made of.
You will use IDS as a shared structural language, then prepare detector crops that we can
recognize in the next two weeks.

## Goals

By the end of the lab, you can:

- explain how an Ideographic Description Sequence (IDS) represents character structure;
- distinguish IDS operators from terminal components;
- convert Unicode Chinese and Nôm text to the recognizer's expanded IDS tokens;
- audit IDS coverage across every page in the durable NomNaOCR sample and locate each gap; and
- export one page of detector boxes as unlabeled crops for recognition.

One important distinction: this is **text-level IDS tokenization**, not visual radical
segmentation. You are not expected to draw boxes around radicals—the supplied materials do
not include those boxes or a model that produces them. Instead, the recognizer will learn IDS
tokens from a complete character crop.

## Required reading

1. Tomohiko Morioka, [“CHISE: Character Processing Based on Character Ontology”](https://link.springer.com/chapter/10.1007/978-3-540-78159-2_14), LKR 2008, pp. 148–162.
2. Jianshu Zhang, Yixing Zhu, Jun Du, and Lirong Dai, [“Radical analysis network for zero-shot learning in printed Chinese character recognition”](https://arxiv.org/abs/1711.01889), ICME 2018.

As you read, focus on two questions: Why are coded character sets alone not enough for
historical characters? And how can a shared structural vocabulary help recognize more classes
than treating every character as an entirely separate label?

## Ground-truth resources

For the exercises, use the two ground-truth files bundled in
[`nom-ids/`](../../nom-ids/). Their order is part of the recognizer setup, so leave the files
as they are:

- `ids_exp.txt` maps one Unicode character to its recursively expanded IDS;
- `vocab_ids.txt` gives the exact checkpoint-sensitive token order.

When you build the reverse lookup, let one IDS map to a list of candidate characters. Different
Unicode characters can share an expanded sequence; a simple `{IDS: character}` dictionary
would quietly lose some of those possibilities.

## 1. What is an IDS?

An IDS is a prefix expression. An Ideographic Description Character (IDC) states a spatial
relationship and is followed by its operands:

- binary examples: `⿰` left–right, `⿱` above–below, `⿴` surround, `⿻` overlap;
- ternary examples: `⿲` left–middle–right and `⿳` above–middle–below.

For example, `明` has direct IDS `⿰日月`: place `日` left of `月`. Nested structures remain
prefix expressions. “Radical vocabulary” is useful shorthand, but the model vocabulary also
contains IDCs, strokes, components, and terminal ideographs that are not radicals in the
strict linguistic sense.

In this course, you will work with **expanded IDS**, which recursively replaces decomposable
components. It is the sequence format used as recognizer ground truth.

## 2. Use the bundled resources

The files already live under `nom-ids/` in the course repository, so there is nothing to
download. Treat them as versioned course inputs: inspect and use them in place, but do not
copy, edit, merge, or reorder them in `$LAB_WORKSPACE`.

## 3. Run the guided notebook

Open [ids_and_nomnaocr_crops.ipynb](ids_and_nomnaocr_crops.ipynb). It is a runnable guide that
walks you through:

1. direct versus expanded IDS;
2. the exact expected conversions for `休好字明林`;
3. Unicode-code-point splitting of a line transcription;
4. corpus-wide IDS coverage and per-page OOV locations across the durable Week 1 sample;
5. manual verification of at least five covered Nôm characters with zi.tools; and
6. a contact sheet of the detector crops exported below.

Reuse the provided `text_to_ids` converter in your team’s separate project repository. Its
structured output preserves character boundaries: rather than flattening a line into one long
string, it retains which IDS belongs to each Unicode code point and includes a direct zi.tools
lookup link for that character.

zi.tools may show a direct IDS, a glyph-region variant, or another normalization. Those
differences are useful observations—record them rather than changing the course dictionary to
force a match. Use zi.tools as a manual reference, not as an automated dependency.

## 4. Audit the NomNaOCR sample, then analyze one page

The notebook scans every durable Week 1 page in `$LAB_WORKSPACE/datasets/nomnaocr-sample`,
parses every matching `line_labels/<relative-path>.txt`, and reports overall IDS coverage plus
every missing character/page pair. Each row gives the character's occurrence count on that page,
the image path, and the corresponding label path, so you can inspect the source directly. Treat
the occurrence count as evidence of the automated-labeling opportunity: it shows how many text
positions cannot yet receive an IDS target from the current vocabulary.

Then choose one of those pages by setting `WEEK05_PAGE` to its relative image path. The selected
page remains the one used for the manual zi.tools checks and unlabeled crop exercise.

A line label contains a polygon and a line transcription, but it does **not** identify the box
that belongs to each character. Do not assign transcription characters directly to detector
boxes. Missed, duplicate, split, or merged detections make positional pairing
unreliable; sequence alignment will address that challenge in Weeks 9–11.

## 5. Export one page of unlabeled inference crops

Run your stronger Week 3 or Week 4 detector on the selected page. Choose a new run name so its
non-overwrite rule keeps earlier predictions safe:

```bash
uv run python -m nom_ocr_lab.run_detector predict \
  --source "$LAB_WORKSPACE/datasets/nomnaocr-sample/images/<relative-page>.jpg" \
  --model "$LAB_WORKSPACE/models/week04-nakagawa/weights/best.pt" \
  --name week05-one-page --imgsz 1280 --batch 1 --device 0
```

Create a small crop-export script in your team repository. Give it a page image and the
detector's YOLO label file. The label file may contain either five values per detection
(`class x_center y_center width height`) or a sixth confidence value from prediction output.

For each detected box, make a slightly larger square crop (scale it by `1.2` and keep it inside
the page), then save it as an RGB PNG. Also write a JSON or TSV manifest that records the crop's
`box_id`, original normalized box, pixel box, final crop coordinates, and confidence when one is
available. Do not assign a character label to a crop yet: matching boxes to transcription
characters is later work. Week 6 will handle the recognizer's image preprocessing.

Use a short interactive CPU allocation for the notebook. Submit detector inference through the
same Slurm workflow you used in Weeks 3–4; dictionary conversion and crop visualization do not
need a GPU.

## Tasks

- [ ] Read both assigned papers and explain direct versus expanded IDS.
- [ ] Inspect the bundled `ids_exp.txt` and `vocab_ids.txt` resources without modifying them.
- [ ] Reproduce the five exact Chinese reference conversions.
- [ ] Reuse text→IDS conversion in the team project repository.
- [ ] Produce a corpus-wide IDS coverage report with every missing character/page location.
- [ ] Record five zi.tools checks, including any normalization or glyph-variant difference.
- [ ] Implement the crop exporter, run the Week 3–4 detector on that page, and inspect its crop contact sheet.
- [ ] Explain why the crops remain unlabeled until sequence alignment.

## Deliverable

A validated IDS converter; a corpus-wide IDS coverage/OOV report; a five-row zi.tools
verification table; and unlabeled recognizer inference crops with their manifest and contact
sheet. Keep generated predictions and crops out of Git.
