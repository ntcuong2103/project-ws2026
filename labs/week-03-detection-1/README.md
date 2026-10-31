# Week 3 — Organizing and Fine-Tuning a Character Detector

[← Back to course README](../../README.md) · [Slurm guide](../../SLURM-QUICK-GUIDE.md) · [Week 4 →](../week-04-detection-2/)

This is a guided introduction to single-class character detection. You will organize a small
YOLO dataset yourself, inspect its labels, then fine-tune YOLO11n on the complete course data.
The detector answers **where is a character?**; recognition later answers **which character?**

The course uses MTH1200 for training/model selection and keeps all of MTH1000 for testing.
This is a teaching protocol for studying transfer to a different collection, not the canonical
MTHv2 research split and not a full paper-replication exercise.

## Data-processing workflow

In this lab, we will create a sample of 24 images from MTH1000 in our dataset workspace, then
arrange and process them there.

## Outcomes

By the end of the lab, your team can:

1. organize data following the `images/{train,val,test}` and `labels/{train,val,test}` convention;
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
directory.

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
testing.** Record that assignment in your team notes before you copy anything. Every manifest
entry must appear exactly once; all 24 pages are MTH1200 pages. The staged sample is deliberately
raw and is **not** a finished dataset:

```text
week03-source-sample/
├── images/
│   └── <page>.jpg           # 24 selected raw MTH1200 pages
├── labels/
│   └── <page>.txt           # matching raw character labels
└── manifest.json            # records the 24 pages students must use exactly once
```

Construct this dataset yourself, preserving each manifest-relative path:

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

Every image and label must have the same relative path and filename stem. Create a `data.yaml`
that identifies the `mth-practice` root, its `images/train`, `images/val`, and `images/test`
directories, and the single `character` class with ID `0`.

For each selected manifest entry, perform this mapping. `relative/page.jpg` is illustrative;
use the actual path in the manifest, including any subdirectories.

| Raw staged input | Destination for a `train` assignment |
| --- | --- |
| `week03-source-sample/images/relative/page.jpg` | `mth-practice/images/train/relative/page.jpg` |
| `week03-source-sample/labels/relative/page.txt` | read this raw file; write its converted rows to `mth-practice/labels/train/relative/page.txt` |

Use the analogous `val` or `test` directory for those assignments. Copy (or symlink) only the
image. Do **not** copy a raw label into `mth-practice/labels`: its pixel-coordinate rows are not
YOLO labels. Keep the staged source intact so you can compare your output with the original.

## 3. Convert the raw labels

For an image of width `W` and height `H`, convert `(x1,y1,x2,y2)` with:

```text
x_center = (x1 + x2) / (2W)       width  = (x2 - x1) / W
y_center = (y1 + y2) / (2H)       height = (y2 - y1) / H
```

The output row is `0 x_center y_center width height`. The first field in a raw MTH row is the annotated character itself (for example, `經 2068.0 1533.0 2261.0 1693.0`) so we will replace it with class `0` given that this is a one-class localization task.

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

Calculate the first box of the first manifest entry by hand. Then implement the conversion
yourself for every selected page. Your code must read the matching image dimensions, write one
normalized class-`0` YOLO row per raw label row, and place that output beside its image in the
chosen `train`, `val`, or `test` split.

Before running the validator, make these checks yourself:

1. Your chosen split counts are exactly 18/2/4, and the 24 image paths are unique.
2. Each converted label has the same relative path and stem as its image.
3. Every output row has five fields, starts with `0`, and contains normalized values rather than
   pixel coordinates.
4. `data.yaml` names this `mth-practice` directory and its three `images/` split directories.

After organizing and converting all 24 pages, validate the result:

```bash
uv run python -m nom_ocr_lab.prepare_detection_data validate-mth-practice \
  --dataset "$LAB_WORKSPACE/datasets/mth-practice"
```

Fix every reported mismatch before continuing. The validator checks the required 18/2/4
assignment, paired stems, normalized class-0 boxes, `data.yaml`, and split membership.

## Smoke-job guide

Use [practice-smoke.sbatch](practice-smoke.sbatch) as a guide for the Slurm directives and the
training/evaluation command structure. It runs against your small practice dataset, so you may
submit it after the validator succeeds. Read and understand the file before running it.

The smoke job requests 2 CPUs, 8 GB RAM, and one shared `a100_2g.10gb_mps` slice. This is the
lightweight default for the 24-page check. If an MPS job fails after it starts, inspect its log
first. When the failure is attributable to the shared MPS allocation rather than your dataset or
command, change only the job's GPU request to the smallest exclusive slice,
`--gres=gpu:a100_2g.10gb:1`, then resubmit. The exclusive slice can take longer to queue; do not
request a full A100 or the larger `a100_3g.20gb` slice for this check.

```bash
sbatch \
  --output="$LAB_WORKSPACE/logs/%x-%j.out" \
  --error="$LAB_WORKSPACE/logs/%x-%j.err" \
  labs/week-03-detection-1/practice-smoke.sbatch
```

This runs ten `imgsz=640` epochs and evaluates the four practice test pages. It is a plumbing
check, not a meaningful accuracy estimate and not a substitute for the required full run. Write
your own full-data `train.sbatch` and `evaluate.sbatch`; adapt the guide rather than submitting
it as either required job.

## 4. Use the complete course dataset

Course staff prepare one canonical full-data view for everyone:

```bash
export MTH_FULL_DATA=/data/shared/project2026/mth-full-project-ws2026
test -f "$MTH_FULL_DATA/data.yaml" && test -f "$MTH_FULL_DATA/manifest.json"
```

Read `$MTH_FULL_DATA/manifest.json` and confirm its 1,080 MTH1200 training pages, 120 MTH1200
validation pages, and 1,000 MTH1000 test pages. Use `$MTH_FULL_DATA/data.yaml` in the jobs your
team writes; `mth-practice/data.yaml` is not the full-training input. MTH1000 is an untouched
held-out test set: never select hyperparameters or checkpoints from its metrics.

## 5. Train through Slurm

Write and submit your own `train.sbatch`. It must start from COCO-pretrained `yolo11n.pt`, use
`$MTH_FULL_DATA/data.yaml`, write every model and log to your group workspace, and select
`best.pt` using MTH1200 validation performance only. Include the job's resource request, exact
training command, and any non-default hyperparameters in your submission.

One successful training run is required. Optional extra runs may change one setting at a time,
for example `imgsz`, learning rate, optimizer, augmentation, or patience. Compare optional runs
using validation results only and give every run a unique `--name`.

Inspect:

- `results.csv` and `results.png` for training progress;
- `box_loss`, `cls_loss`, and `dfl_loss`;
- `weights/best.pt`, selected by validation performance; and
- `weights/last.pt`, the final epoch rather than automatically the best model.

Reduce `--batch` if the job runs out of GPU memory. Do not silently change the dataset split.

## 6. Understand the metrics before evaluating

Your `evaluate.sbatch` reports detection metrics, so you must be able to explain them before
you run it. The evaluator first sorts predictions by confidence. At an IoU threshold `τ`, a
prediction is a true positive (TP) only when it matches one previously unmatched ground-truth
box with IoU at least `τ`. An unmatched prediction is a false positive (FP); an unmatched
ground-truth box is a false negative (FN).

### Intersection over Union (IoU)

For prediction box `P` and ground-truth box `G`, IoU measures localization overlap:

```text
IoU(P, G) = area(P ∩ G) / area(P ∪ G)
          = intersection_area / (area(P) + area(G) - intersection_area)
```

Calculate IoU for identical boxes, disjoint boxes, and the pair `(100,100,150,150)` versus
`(120,120,170,170)`. For the last pair, the result is
`900 / (2500 + 2500 - 900) ≈ 0.2195`.

- **Precision:** fraction of predicted boxes that match a character.
- **Recall:** fraction of ground-truth characters detected.
- **mAP50:** average precision at IoU 0.50.
- **mAP50–95:** average precision across IoU 0.50 through 0.95.

### Precision, recall, and average precision

At one confidence cutoff and IoU threshold `τ`:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

Precision answers: “Of the boxes the detector emitted, how many were correct?” It falls when
the detector produces extra, duplicate, or badly localized boxes. Recall answers: “Of the
annotated characters, how many did the detector find?” It falls when the detector misses
characters. Changing the confidence cutoff trades recall against precision: accepting more
low-confidence boxes can find missed characters but also adds false positives.

Average precision (AP) summarizes the complete precision–recall curve rather than selecting one
confidence cutoff. Sort predictions by confidence, calculate precision and recall after each
additional prediction, then compute the interpolated area under that curve at IoU threshold `τ`:

```text
p_interp(r, τ) = max precision at any recall r' ≥ r
APτ = ∫ from 0 to 1 of p_interp(r, τ) dr
```

AP rewards a detector that reaches high recall while keeping high-confidence predictions
precise; it is not the same as precision at one selected confidence cutoff.

### mAP50 and mAP50–95

Mean average precision (mAP) averages AP across classes and, when specified, IoU thresholds.
This course has one class (`character`), so the class mean does not change the value:

```text
mAP50    = AP at τ = 0.50
mAP50–95 = (AP0.50 + AP0.55 + AP0.60 + ... + AP0.95) / 10
```

mAP50 counts a prediction as matched when IoU is at least 0.50, so it mainly shows whether
characters are found in roughly the right place. mAP50–95 averages ten increasingly strict IoU
thresholds from 0.50 to 0.95, penalizing loose or shifted boxes more strongly.

High mAP50 with lower mAP50–95 often means characters are found but some boxes are loose or
shifted. For dense OCR pages, recall is especially important because a missed box cannot be
recovered by the downstream recognizer.

## 7. Compare before and after on MTH1000

After completing the metric exercise, write and submit your own `evaluate.sbatch`. Evaluate
untouched `yolo11n.pt` and the trained `best.pt` on the `test` split of
`$MTH_FULL_DATA/data.yaml` with identical settings, storing both outputs in your group workspace.
Do not alter the shared dataset or use MTH1000 results to select a model. Interpret the MTH1000
comparison rather than aiming for a prescribed score.

## Deliverables

1. The organized practice dataset, split-assignment record, successful validator output, and one
   manual box calculation.
2. Confirmation of the shared full-data manifest's MTH1200 train/validation and MTH1000 test counts.
3. Your training and evaluation `sbatch` files and Slurm stdout/stderr.
4. `results.csv`, training plots, `best.pt`, and `last.pt`.
5. COCO-before versus MTH-after test metrics, three IoU calculations, and a short interpretation.

Grading emphasizes correct structure, workflow, artifacts, and interpretation—not a target
metric. Do not commit datasets, weights, logs, metrics, predictions, or plots.
