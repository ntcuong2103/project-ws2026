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
export LAB_REPO=/home/projectgX/project-ws2026
export LAB_WORKSPACE=/data/projectgX/lab-exercise
cd "$LAB_REPO"
uv sync
mkdir -p "$LAB_WORKSPACE/logs" "$LAB_WORKSPACE/models/pretrained"
```

Keep code under `/home/projectgX` and generated data, models, logs, and metrics under
`$LAB_WORKSPACE`. `/data/shared/project2026` is read-only. Full training and evaluation must
use `sbatch`, not a long-running interactive `srun` session.

### Download and verify the YOLO11n starting weights

The Slurm jobs expect the COCO-pretrained checkpoint at
`$LAB_WORKSPACE/models/pretrained/yolo11n.pt`. Download it once from a shell with internet
access. Running from the pretrained-model directory keeps the generated weight out of the Git
repository:

```bash
cd "$LAB_WORKSPACE/models/pretrained"

"$LAB_REPO/.venv/bin/python" -c \
  'from ultralytics import YOLO; YOLO("yolo11n.pt")'

cd "$LAB_REPO"
```

Check both that the file is present and that Ultralytics can load it:

```bash
test -s "$LAB_WORKSPACE/models/pretrained/yolo11n.pt" \
  && echo "yolo11n.pt is present"

"$LAB_REPO/.venv/bin/python" -c \
  'import os; from ultralytics import YOLO; p=os.environ["LAB_WORKSPACE"] + "/models/pretrained/yolo11n.pt"; YOLO(p); print("yolo11n.pt is ready")'
```

Do not submit a training job unless the final command prints `yolo11n.pt is ready`. If the
compute server cannot access the internet, copy an existing `yolo11n.pt` into the expected
directory. Alternatively, export `YOLO11N_WEIGHTS=/absolute/path/to/yolo11n.pt` before
submitting; the supplied Week 3 Slurm jobs honor that override.

## 2. Stage the raw practice sample

The command below copies a fixed sample into your data area. It does **not** construct the
finished YOLO dataset. The sample contains 24 MTH1200 pages with raw
`label x1 y1 x2 y2` annotations.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data stage-mth-practice \
  --source /data/shared/project2026/tkh-mth2k2/MTH1200
```

Inspect `datasets/week03-source-sample/manifest.json`; it lists the 24 sampled page/label pairs.
**Students choose which 18 pages go to training, which 2 go to validation, and which 4 go to
testing.**
The staged sample is deliberately raw and is **not** a finished dataset:

```text
week03-source-sample/
├── images/
│   └── <page>.jpg           # 24 selected raw MTH1200 pages
├── labels/
│   └── <page>.txt           # matching raw character labels
└── manifest.json            # records the 24 pages students must use exactly once
```

Construct this dataset yourself, preserving each page's relative path:

```text
mth-practice/
├── images/
│   ├── train/    # 18 MTH1200 pages
│   ├── val/      # 2 MTH1200 pages
│   └── test/     # 4 MTH1200 pages
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

### Raw MTH label field

The first field in a raw MTH row may be a numeric class ID or the annotated character itself
(for example, `經 2068.0 1533.0 2261.0 1693.0`). This is expected. The supplied
`convert-mth-label` helper ignores that first field and converts the four numeric coordinates to
one-class YOLO labels. **Students do not need to edit, map, or remove character labels.** If the
helper reports that a coordinate field is non-numeric or a box is outside its image, report that
specific source file to the teaching team.

### Why the coordinates are normalized

Raw annotations are measured in pixels, so their values depend on the page resolution. For
example, an `x` coordinate of `400` means the horizontal midpoint of an 800-pixel-wide image,
but it is only one quarter of the way across a 1600-pixel-wide image. A detector is trained on
pages that may have different widths and heights and may later resize them for training. Using
pixel coordinates directly would make the same relative character position have different target
values solely because the image was scanned or resized differently.

YOLO therefore stores each coordinate as a fraction of the image dimension: divide horizontal
values (`x_center` and `width`) by `W`, and vertical values (`y_center` and `height`) by `H`.
The resulting values are normally in the range `0` to `1`, describing a position and size
relative to that particular page. Thus, `x_center = 0.5` means “centered horizontally,” whatever
the image width is; `width = 0.1` means “one tenth of the image width.” This makes the label
resolution-independent and lets the training software correctly transform the boxes when it
resizes or augments an image.

For example, in an 800 × 1000 pixel image, a box from `(x1, y1, x2, y2) = (120, 200, 280, 500)`
has center `(200, 350)` and size `(160, 300)` pixels. Its YOLO row is:

```text
0 0.250000 0.350000 0.200000 0.300000
```

Always use the original image's width and height for this calculation, keep the order
`x_center y_center width height`, and do not divide every value by the same dimension. Values
outside `0`–`1` usually indicate an invalid box or that the wrong image dimensions were used.

Calculate the first box of the first manifest entry by hand. Then convert each file with the
provided one-file helper while placing it in the corresponding output split:

```bash
uv run python -m nom_ocr_lab.prepare_detection_data convert-mth-label \
  --image "$LAB_WORKSPACE/datasets/week03-source-sample/images/<page>.jpg" \
  --label "$LAB_WORKSPACE/datasets/week03-source-sample/labels/<page>.txt" \
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
