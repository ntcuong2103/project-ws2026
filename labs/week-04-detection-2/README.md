# Week 4 — Fine-Tuning on Nôm and Reading Detection Metrics

[← Week 3](../week-03-detection-1/) · [Back to course README](../../README.md) · [Week 5 →](../week-05-recognition-1/)

This guided lab adapts the Week 3 character detector to a small Nôm collection. You will use
the supplied Nakagawa splits, interpret standard detection metrics, and compare the detector
before and after adaptation on an external NomNaOCR reference set. This is an introduction to
the workflow, not a complete reproduction of a research experiment.

## Outcomes

By the end of the lab, your team can:

1. explain why thousands of character identities become one detection class;
2. fine-tune from a previous checkpoint without leaking validation or test data;
3. interpret IoU, precision, recall, mAP50, and mAP50–95; and
4. make a qualitative before/after comparison on the same external data.

## 1. Verify the Week 3 handoff

```bash
export LAB_WORKSPACE=/data/projectgX/lab-exercise
test -f "$LAB_WORKSPACE/models/week03-mth1200/weights/best.pt"
test -f "$LAB_WORKSPACE/models/week03-mth1200/weights/last.pt"
mkdir -p "$LAB_WORKSPACE/logs"
```

Do not substitute a checkpoint selected using MTH1000 test results.

## 2. Prepare Nakagawa

Nakagawa contains many character identities, but the detector only learns the shared geometric
task “where is a character?” Recognition models later learn “which character is it?” The helper
uses the supplied `train.txt`, `val.txt`, and `test.txt` manifests and rewrites every valid class
ID to `0` while preserving its normalized box coordinates.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data prepare-nakagawa \
  --source /data/shared/project2026/nakagawalab
```

Inspect three source rows and their generated counterparts under
`datasets/nakagawa-character-detection/labels/`. Confirm that the class ID changes to `0` and
the four coordinates do not change. Read the generated manifest and verify that the three
supplied splits are disjoint.

## 3. Fine-tune through Slurm

The supplied [train.sbatch](train.sbatch) starts from the Week 3 `best.pt`, trains only on
Nakagawa train, and uses Nakagawa validation for early stopping and checkpoint selection.

```bash
sbatch \
  --output="$LAB_WORKSPACE/logs/%x-%j.out" \
  --error="$LAB_WORKSPACE/logs/%x-%j.err" \
  labs/week-04-detection-2/train.sbatch
```

Inspect the Slurm log, `results.csv`, plots, `weights/best.pt`, and `weights/last.pt`. The final
Nakagawa test set is for evaluation, not model selection.

## 4. Understand the metrics

For prediction box `P` and ground-truth box `G`:

```text
IoU(P,G) = area(P intersection G) / area(P union G)
```

Calculate IoU for identical boxes, disjoint boxes, and the pair `(100,100,150,150)` versus
`(120,120,170,170)`. For the last pair, the result is
`900 / (2500 + 2500 - 900) ≈ 0.2195`.

- **Precision:** among predicted boxes, the fraction matched to a character.
- **Recall:** among ground-truth characters, the fraction successfully detected.
- **mAP50:** average precision when a match requires IoU at least 0.50.
- **mAP50–95:** average precision over IoU thresholds 0.50 through 0.95.

High mAP50 with lower mAP50–95 often means characters are found but some boxes are loose or
shifted. For dense OCR pages, recall is especially important because a missed box cannot be
recovered by the downstream recognizer.

## 5. Prepare the external NomNaOCR reference set

The server provides the smaller Roboflow export as a ZIP. The helper copies every labeled page
from all archive partitions into one external test view. The original archive partition remains
in each relative path so equal filenames cannot collide.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data prepare-nomnaocr-archive \
  --archive /data/shared/project2026/nom-detection-val.v2i.yolov11.zip
```

Inspect `datasets/nomnaocr-character-detection/manifest.json` and `data.yaml`. Do not use this
external set for early stopping or hyperparameter selection.

## 6. Evaluate before and after

Submit [evaluate.sbatch](evaluate.sbatch). It evaluates the Nakagawa checkpoint on Nakagawa
test, evaluates the Week 3 and Week 4 checkpoints on identical NomNaOCR data, and produces the
Week 4 NomNaOCR prediction handoff used by later recognition labs.

```bash
sbatch \
  --output="$LAB_WORKSPACE/logs/%x-%j.out" \
  --error="$LAB_WORKSPACE/logs/%x-%j.err" \
  labs/week-04-detection-2/evaluate.sbatch
```

Compare the two NomNaOCR result files qualitatively. Discuss whether Nôm fine-tuning changed
precision, recall, and strict localization quality, and inspect a few missed or poorly localized
characters. Possible traits include small glyphs, touching characters, low contrast, damage, or
page edges.

> **Reference only — not a reproduction target:** a reported detector result gives Precision
> `0.9875`, Recall `0.9922`, mAP50 `0.9867`, and mAP50–95 `0.9674`. Dataset version, splitting,
> preprocessing, training, and evaluator settings may differ. Do not expect your values to match
> and do not tune against these numbers.

## Deliverables

1. Three before/after Nakagawa label rows and an explanation of the one-class conversion.
2. The Week 4 training `sbatch` file, Slurm logs, curves, `best.pt`, and `last.pt`.
3. Nakagawa held-out test metrics.
4. Week 3 versus Week 4 NomNaOCR metrics and a short qualitative interpretation.
5. The complete `predictions/week04-nomnaocr/` handoff and several inspected error examples.

Grading emphasizes correct workflow, artifact organization, metric understanding, and thoughtful
interpretation—not reproducing the reference values. Do not commit generated datasets, weights,
logs, metrics, predictions, or plots.
