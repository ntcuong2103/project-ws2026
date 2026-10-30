# Week 4 — Character-Detector Fine-Tuning and Error Analysis

[← Week 3](../week-03-detection-1/) · [Back to course README](../../README.md) · [Week 5 →](../week-05-recognition-1/)

> **Interactive Slurm reminder:** Whenever you use `srun --pty ...`, run `exit` as soon as you finish. This releases the allocation; do not leave an interactive session idle.

Week 4 starts from the MTH1000 `best.pt` produced in Week 3. You will fine-tune it on
TUAT-Nakagawa, calculate character-level detection metrics, and determine whether adaptation
improves NomNaOCR. Do not restart from COCO weights.

## Outcomes

By the end of Week 4, your team will:

1. explain why Nakagawa character identities are collapsed into one detection class;
2. fine-tune and select a model without leaking the test split;
3. calculate IoU, precision, recall, AP, mAP50, and mAP50–95;
4. compare an independent saved-label evaluation with Ultralytics validation;
5. compare Week 3 and Week 4 checkpoints on identical NomNaOCR data; and
6. classify character-level detection errors.

## 1. Verify the Week 3 handoff

```bash
test -f "$LAB_WORKSPACE/models/week03-mth1000/weights/best.pt"
test -f "$LAB_WORKSPACE/metrics/week03-nomnaocr/metrics.json"
test -f "$LAB_WORKSPACE/datasets/nomnaocr-character-detection/data.yaml"
```

If one is missing, complete Week 3 rather than substituting a different checkpoint or split.

## 2. Why Nakagawa becomes a single-class dataset

Inspect the original mapping and count its entries:

```bash
head -5 /data/shared/project2026/nakagawalab/class_mapping.txt
wc -l /data/shared/project2026/nakagawalab/class_mapping.txt
```

The mapping contains approximately 7,504 character identities. A small page collection gives
most identities few or no examples, creating extreme class imbalance. Detection only needs to
learn the shared geometric task “where is a character?” Recognition later learns “which
character is it?” from cropped character images and an appropriate recognition model.

Prepare Nakagawa using its fixed `train.txt`, `val.txt`, and `test.txt` manifests. Preparation
preserves those splits and rewrites every identity to class `0`.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data nakagawa \
  --source /data/shared/project2026/nakagawalab
```

Use the first generated manifest entry to compare its `source_label` with the matching file
below the prepared `labels/<split>/` tree. Verify at least three rows: the leading IDs must
change to `0`, while `x_center`, `y_center`, `width`, and `height` remain numerically identical.
Confirm that all generated rows start with `0`.

## 3. Fine-tune and evaluate Nakagawa

Train only on Nakagawa train, use validation for early stopping and `best.pt` selection, and
inspect the held-out test metrics once after settings are fixed.

```bash
uv run python -m nom_ocr_lab.run_detector train \
  --data "$LAB_WORKSPACE/datasets/nakagawa-character-detection/data.yaml" \
  --model "$LAB_WORKSPACE/models/week03-mth1000/weights/best.pt" \
  --name week04-nakagawa --epochs 100 --imgsz 1280 \
  --batch 8 --patience 10 --device 0

uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/nakagawa-character-detection/data.yaml" \
  --model "$LAB_WORKSPACE/models/week04-nakagawa/weights/best.pt" \
  --split test --name week04-nakagawa-test \
  --imgsz 1280 --batch 8 --device 0
```

## 4. IoU by hand

For prediction `P` and ground-truth character box `G`:

```text
IoU(P,G) = area(P ∩ G) / area(P ∪ G)
         = intersection / (area(P) + area(G) - intersection)
```

Calculate these before running code:

1. identical boxes `(100,100,150,150)` and `(100,100,150,150)`;
2. disjoint boxes `(100,100,150,150)` and `(200,200,250,250)`; and
3. partial boxes `(100,100,150,150)` and `(120,120,170,170)`.

For the partial case, the intersection is `30 × 30 = 900`, each box area is `2500`, and the
IoU is `900 / (2500 + 2500 - 900) ≈ 0.2195`. Verify all three with
`nom_ocr_lab.detection_visualization.box_iou`.

At a chosen IoU threshold, matching is one-to-one: one prediction can match at most one
ground-truth character, and one ground-truth character can match at most one prediction.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

- **TP:** a predicted box matched at or above the evaluation IoU threshold.
- **FP:** a predicted box without a match.
- **FN:** a ground-truth character without a match.

Lowering the confidence threshold normally retains more predictions, raising recall but often
lowering precision. Sweeping confidence produces a precision–recall curve. AP is its area;
the course's independent evaluator uses 101-point interpolated AP. With one class, mAP equals
the AP of `character`.

- **mAP50:** AP at evaluation IoU `0.50`.
- **mAP50–95:** AP averaged at `0.50, 0.55, …, 0.95`.

High mAP50 with lower mAP50–95 suggests that characters are found but boxes are loose or
shifted.

## 5. Re-evaluate NomNaOCR

Use exactly the Week 3 NomNaOCR split, `imgsz`, batch, and `max_det` settings so the checkpoint
is the intended experimental difference.

```bash
uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/nomnaocr-character-detection/data.yaml" \
  --model "$LAB_WORKSPACE/models/week04-nakagawa/weights/best.pt" \
  --split test --name week04-nomnaocr \
  --imgsz 1280 --batch 8 --device 0

uv run python -m nom_ocr_lab.run_detector predict \
  --source "$LAB_WORKSPACE/datasets/nomnaocr-character-detection/images/test" \
  --model "$LAB_WORKSPACE/models/week04-nakagawa/weights/best.pt" \
  --name week04-nomnaocr --imgsz 1280 --batch 8 --device 0

uv run python -m nom_ocr_lab.compare_detection_metrics \
  --week3 "$LAB_WORKSPACE/metrics/week03-nomnaocr/metrics.json" \
  --week4 "$LAB_WORKSPACE/metrics/week04-nomnaocr/metrics.json" \
  --output "$LAB_WORKSPACE/metrics/nomnaocr-week03-vs-week04"
```

This complete run is the Week 6–7 recognition handoff. Keep
`$LAB_WORKSPACE/predictions/week04-nomnaocr/manifest.json` together with its nested `labels/`
tree. The command writes one label file for every input page (empty if no character is detected),
preserves the relative page path, and records the model and prediction settings. Do not move or
renumber these labels: recognition uses the relative page path as `page_id` and row order as
`box_id`.

## 6. Independent saved-label evaluation

Ultralytics `model.val()` above is the authoritative course metric. Now reconstruct the metric
from saved predictions to understand the calculation. Generate a second prediction set with a
low confidence cutoff so the precision–recall sweep retains low-confidence candidates:

```bash
uv run python -m nom_ocr_lab.run_detector predict \
  --source "$LAB_WORKSPACE/datasets/nomnaocr-character-detection/images/test" \
  --model "$LAB_WORKSPACE/models/week04-nakagawa/weights/best.pt" \
  --name week04-nomnaocr-pr --imgsz 1280 --batch 8 --device 0 \
  --conf 0.001 --iou 0.5

uv run python -m nom_ocr_lab.evaluate_detection_predictions \
  --dataset "$LAB_WORKSPACE/datasets/nomnaocr-character-detection" \
  --predictions "$LAB_WORKSPACE/predictions/week04-nomnaocr-pr/labels" \
  --split test \
  --output "$LAB_WORKSPACE/metrics/week04-nomnaocr-saved-labels" \
  --iou-thresholds 0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95
```

Do not confuse the two meanings of IoU:

- prediction `--iou` is the NMS overlap threshold used to suppress duplicate predictions;
- `--iou-thresholds` defines the overlaps required to match predictions to ground truth.

Compare the independent `metrics.json` with
`$LAB_WORKSPACE/metrics/week04-nomnaocr/metrics.json`. Small differences are expected because
Ultralytics controls preprocessing, NMS, maximum detections, confidence filtering, and metric
interpolation internally. Report Ultralytics validation as the official number and use the
independent result to explain the mechanics.

## 7. Character-level error gallery

```bash
uv run python -m nom_ocr_lab.build_detection_gallery \
  --dataset "$LAB_WORKSPACE/datasets/nomnaocr-character-detection" \
  --predictions "$LAB_WORKSPACE/predictions/week04-nomnaocr/labels" \
  --output "$LAB_WORKSPACE/predictions/week04-nomnaocr-error-gallery" \
  --split test --selection errors --iou 0.5 --limit 10
```

Ground truth is green and predictions are red. Review at least ten errors across five pages:

- **Missed:** a ground-truth character has no matched prediction.
- **Spurious:** a prediction has no matching character.
- **Merge:** one predicted box covers several characters.
- **Split:** several predicted boxes cover one character.
- **Poor localization:** the correct character is targeted but overlap is insufficient.

Automatically reported merge and split counts are candidates for manual confirmation. Do not
analyze Unicode identity: this detector predicts only the `character` class.

## 8. Wrap-up questions

1. Inspect three false negatives. Do they share a trait such as small size, touching glyphs,
   low contrast, damage, or proximity to a page edge?
2. Using the Week 3 controlled experiment, did 1280 improve mAP50–95 more than mAP50? What
   does that suggest about localization?
3. What would happen to loss and per-class metrics if `data.yaml` used `nc: 7504` with the
   original Nakagawa identities? Explain without running the expensive experiment.
4. Find a prediction whose nearest ground truth has IoU between 0.30 and 0.50. Does it count
   at the mAP50 threshold? How does it affect mAP50–95?

## Week 4 deliverables

1. Before/after Nakagawa label comparison.
2. Nakagawa-fine-tuned `best.pt` and held-out test metrics.
3. Week 4 NomNaOCR metrics and nested predictions.
4. Complete NomNaOCR detector handoff: nested labels plus `manifest.json` for every page.
5. Week 3 versus Week 4 Markdown/CSV comparison table.
6. Independent evaluator JSON/Markdown and a short comparison with `model.val()`.
7. Ten-image error gallery and a half-page analysis answering the wrap-up questions.

Do not commit prepared datasets, weights, predictions, plots, caches, or logs.
