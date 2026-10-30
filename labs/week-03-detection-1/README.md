# Week 3 — Organizing and Fine-Tuning a Character Detector

[← Back to course README](../../README.md) · [Slurm guide](../../SLURM-QUICK-GUIDE.md) · [Week 4 →](../week-04-detection-2/)

This is a guided introduction to single-class character detection. You will organize a small
YOLO dataset yourself, inspect its labels, then fine-tune YOLO11n on the complete course data.
The detector answers **where is a character?**; recognition later answers **which character?**

The course uses MTH1200 for training/model selection and keeps all of MTH1000 for testing.
This is a teaching protocol for studying transfer to a different collection, not the canonical
MTHv2 research split and not a full paper-replication exercise.

## Outcomes

By the end of the lab, your team can:

1. explain the `images/{train,val,test}` and `labels/{train,val,test}` convention;
2. convert pixel character boxes to normalized, single-class YOLO labels;
3. submit reproducible training and evaluation jobs with `sbatch`;
4. distinguish training loss, validation model selection, and held-out testing; and
5. interpret `best.pt`, `last.pt`, precision, recall, mAP50, and mAP50–95.

## 1. Environment and data boundaries

```bash
export LAB_WORKSPACE=/data/projectgX/lab-exercise
uv sync
mkdir -p "$LAB_WORKSPACE/logs" "$LAB_WORKSPACE/models/pretrained"
```

Keep code under `/home/projectgX` and generated data, models, logs, and metrics under
`$LAB_WORKSPACE`. `/data/shared/project2026` is read-only. Full training and evaluation must
use `sbatch`, not a long-running interactive `srun` session.

The Slurm jobs expect the COCO checkpoint at
`$LAB_WORKSPACE/models/pretrained/yolo11n.pt`. An instructor can instead export
`YOLO11N_WEIGHTS` with the absolute path to an existing shared copy.

## 2. Stage the raw practice sample

The command below copies a fixed sample into your data area. It does **not** construct the
finished YOLO dataset. The sample contains 20 MTH1200 pages and 4 MTH1000 pages with raw
`class_id x1 y1 x2 y2` labels.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data stage-mth-practice \
  --train-source /data/shared/project2026/tkh-mth2k2/MTH1200 \
  --test-source /data/shared/project2026/tkh-mth2k2/MTH1000
```

Inspect `datasets/week03-source-sample/manifest.json`. Its `suggested_split` assigns 18
MTH1200 pages to train, 2 MTH1200 pages to validation, and 4 MTH1000 pages to test.

Construct this dataset yourself, preserving each page's relative path:

```text
mth-practice/
├── images/
│   ├── train/    # 18 MTH1200 pages
│   ├── val/      # 2 MTH1200 pages
│   └── test/     # 4 MTH1000 pages
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Every image and label must have the same relative path and filename stem. A valid `data.yaml`
is:

```yaml
path: /data/projectgX/lab-exercise/datasets/mth-practice
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: character
```

## 3. Convert the raw labels

For an image of width `W` and height `H`, convert `(x1,y1,x2,y2)` with:

```text
x_center = (x1 + x2) / (2W)       width  = (x2 - x1) / W
y_center = (y1 + y2) / (2H)       height = (y2 - y1) / H
```

The output row is `0 x_center y_center width height`. The original character identity is
replaced by class `0` because this is a one-class localization task.

Calculate the first box of the first manifest entry by hand. Then convert each file with the
provided one-file helper while placing it in the corresponding output split:

```bash
uv run python -m nom_ocr_lab.prepare_detection_data convert-mth-label \
  --image "$LAB_WORKSPACE/datasets/week03-source-sample/mth1200/images/<page>.jpg" \
  --label "$LAB_WORKSPACE/datasets/week03-source-sample/mth1200/label_char/<page>.txt" \
  --output "$LAB_WORKSPACE/datasets/mth-practice/labels/train/<page>.txt"
```

The helper refuses to overwrite an existing label. After organizing all 24 pages, validate
the result:

```bash
uv run python -m nom_ocr_lab.prepare_detection_data validate-mth-practice \
  --dataset "$LAB_WORKSPACE/datasets/mth-practice"
```

Fix every reported mismatch before continuing. The validator checks the required 18/2/4
assignment, paired stems, normalized class-0 boxes, `data.yaml`, and split membership.

Instructors can smoke-test the complete 24-page path before scheduling the full run:

```bash
sbatch \
  --output="$LAB_WORKSPACE/logs/%x-%j.out" \
  --error="$LAB_WORKSPACE/logs/%x-%j.err" \
  labs/week-03-detection-1/practice-smoke.sbatch
```

This runs ten `imgsz=640` epochs and evaluates the four practice test pages. It is a plumbing
check, not a meaningful accuracy estimate and not a substitute for the required full run.

## 4. Prepare the complete course dataset

The full-data helper automates the repetitive work. With the server data it creates 1,080
MTH1200 training pages, 120 MTH1200 validation pages, and all 1,000 MTH1000 test pages.
Selection is deterministic with seed `2026`; images are symlinked and labels are converted.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data prepare-mth-full \
  --train-source /data/shared/project2026/tkh-mth2k2/MTH1200 \
  --test-source /data/shared/project2026/tkh-mth2k2/MTH1000
```

Read `datasets/mth-course-full/manifest.json` and confirm the split counts. MTH1000 is an
untouched held-out test set: never select hyperparameters or checkpoints from its metrics.

## 5. Train through Slurm

The supplied [train.sbatch](train.sbatch) starts from COCO-pretrained `yolo11n.pt`, trains on
MTH1200, and selects `best.pt` from MTH1200 validation performance.

```bash
sbatch \
  --output="$LAB_WORKSPACE/logs/%x-%j.out" \
  --error="$LAB_WORKSPACE/logs/%x-%j.err" \
  labs/week-03-detection-1/train.sbatch
```

One successful training run is required. Optional extra runs may change one setting at a time,
for example `imgsz`, learning rate, optimizer, augmentation, or patience. Compare optional runs
using validation results only and give every run a unique `--name`.

Inspect:

- `results.csv` and `results.png` for training progress;
- `box_loss`, `cls_loss`, and `dfl_loss`;
- `weights/best.pt`, selected by validation performance; and
- `weights/last.pt`, the final epoch rather than automatically the best model.

Reduce `--batch` if the job runs out of GPU memory. Do not silently change the dataset split.

## 6. Compare before and after on MTH1000

Submit [evaluate.sbatch](evaluate.sbatch) only after fixing the training configuration and
selecting the checkpoint. It evaluates untouched `yolo11n.pt` and the trained `best.pt` with
identical settings on full MTH1000.

```bash
sbatch \
  --output="$LAB_WORKSPACE/logs/%x-%j.out" \
  --error="$LAB_WORKSPACE/logs/%x-%j.err" \
  labs/week-03-detection-1/evaluate.sbatch
```

Interpret the comparison rather than aiming for a prescribed score:

- **Precision:** fraction of predicted boxes that match a character.
- **Recall:** fraction of ground-truth characters detected.
- **mAP50:** average precision at IoU 0.50.
- **mAP50–95:** average precision across IoU 0.50 through 0.95.

## Deliverables

1. The organized practice dataset, successful validator output, and one manual box calculation.
2. Full-data manifest showing MTH1200 train/validation and MTH1000 test counts.
3. The training `sbatch` file and Slurm stdout/stderr.
4. `results.csv`, training plots, `best.pt`, and `last.pt`.
5. COCO-before versus MTH-after test metrics and a short interpretation.

Grading emphasizes correct structure, workflow, artifacts, and interpretation—not a target
metric. Do not commit datasets, weights, logs, metrics, predictions, or plots.
