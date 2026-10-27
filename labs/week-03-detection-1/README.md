# Week 3 — Single-Class Character Detection with MTH1000

[← Back to course README](../../README.md) · [Slurm guide](../../SLURM-QUICK-GUIDE.md) · [Week 4 →](../week-04-detection-2/)

This week has one task: **find every individual character and draw one bounding box around it**.
The detector does not identify characters. Every box therefore has class `0`, named
`character`. Line detection and character recognition are out of scope.

## Outcomes

By the end of Week 3, your team will:

1. read, write, and verify YOLO character boxes by hand;
2. construct a YOLO dataset from MTH1000 `images/` and `label_char/`;
3. run a same-data fitting check on a deterministic 24-page sample;
4. interpret YOLO's losses, validation metrics, and checkpoints;
5. train YOLO11n on a full MTH1000 80/10/10 split; and
6. evaluate and visualize character detections on MTH1000 and NomNaOCR.

The resulting MTH1000 `best.pt` is the required input to Week 4.

## 1. Environment and data boundaries

Complete Week 1 setup first. Code stays under `/home/projectgX`; generated datasets,
checkpoints, predictions, metrics, and logs stay under `/data/projectgX/lab-exercise`.

```bash
export LAB_WORKSPACE=/data/projectgX/lab-exercise
uv sync
```

The shared source tree is read-only. Preparation creates image symlinks and writable labels
under `$LAB_WORKSPACE`; it refuses to write into `/data/shared/project2026` or overwrite a
non-empty destination.

If `import cv2` fails with `libGL.so.1`, replace GUI OpenCV in the active environment:

```bash
uv pip uninstall opencv-python
uv pip install opencv-python-headless
uv run python -c "import cv2; from ultralytics import YOLO; print('ok')"
```

Run training and full-dataset inference through Slurm. Put the commands below inside an
`sbatch` job following the linked guide. `--device 0` means the first GPU visible inside the
allocation; never commit a physical GPU or MIG UUID.

## 2. YOLO dataset anatomy

A prepared detection dataset has three ingredients:

```text
dataset/
├── images/train/.../page.jpg
├── images/val/.../page.jpg
├── images/test/.../page.jpg
├── labels/train/.../page.txt
├── labels/val/.../page.txt
├── labels/test/.../page.txt
└── data.yaml
```

Every image must have a label with the same relative path and filename stem. For example,
`images/train/book/page.jpg` pairs with `labels/train/book/page.txt`. Ultralytics infers the
label path by replacing the `images` path component with `labels`; a wrong directory or stem
can make annotations appear missing.

Each target row is:

```text
0 x_center y_center width height
```

Coordinates are normalized by image width or height. MTH1000 supplies pixel boxes as
`class_id x1 y1 x2 y2`; preparation converts them with:

```text
x_center = (x1 + x2) / (2 × image_width)
y_center = (y1 + y2) / (2 × image_height)
width    = (x2 - x1) / image_width
height   = (y2 - y1) / image_height
```

To recover pixel corners:

```text
x1 = (x_center - width / 2) × image_width
y1 = (y_center - height / 2) × image_height
x2 = (x_center + width / 2) × image_width
y2 = (y_center + height / 2) × image_height
```

The original character identity is discarded. Detection answers “is a character here?”;
recognition in later weeks answers “which character is it?”

`data.yaml` describes the image locations and class vocabulary:

```yaml
path: /absolute/path/to/prepared-dataset
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: character
```

`train`, `val`, and `test` may also point to text manifests of image paths. `nc` must equal
the number of entries in `names`.

## 3. Classroom exercise: fit the same 24 pages

This command deterministically selects 24 matched pages using seed `2026`, converts their
character boxes, and records the exact selection in `manifest.json`. `train`, `val`, and `test`
all reference the same physical pages for this exercise.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data mth1000 \
  --source /data/shared/project2026/tkh-mth2k2/MTH1000 \
  --mode practice
```

### Manual box exercise

Open `manifest.json` and use its first entry. Find that page under the source `images/` and
its `source_label` under MTH1000. Then:

1. obtain the image width and height with Pillow or an image viewer;
2. take the first source row and calculate its normalized YOLO row by hand;
3. find the corresponding generated row under `labels/all/`;
4. convert the generated row back to pixel corners; and
5. confirm that the recovered corners equal the source corners apart from rounding.

Also inspect `data.yaml` and three generated labels. Confirm that every row begins with `0`
and that every image-label pair has an identical relative stem.

### Train and inspect the run

```bash
uv run python -m nom_ocr_lab.run_detector train \
  --data "$LAB_WORKSPACE/datasets/mth1000-practice/data.yaml" \
  --model yolo11n.pt --name week03-practice \
  --epochs 10 --imgsz 640 --batch 8 --patience 0 --device 0

uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/mth1000-practice/data.yaml" \
  --model "$LAB_WORKSPACE/models/week03-practice/weights/best.pt" \
  --split test --name week03-practice-fit \
  --imgsz 640 --batch 8 --device 0
```

`yolo11n.pt` starts from features pretrained on COCO rather than random weights. The course
wrapper always enables `single_cls=True`, so every box is treated as `character`; preparation
also rewrites labels to class `0` so the stored dataset is unambiguous.

- `imgsz` controls the resized/padded input. Small characters retain more detail at 1280 than
  at 640, but training costs more memory and time.
- `epochs` is the maximum number of passes over train.
- `patience` stops training after that many validation epochs without improvement. It is zero
  only for this deliberate fitting check.
- `max_det` caps retained detections per page; the wrapper defaults to 1000 for dense pages.

Read the run output:

- `box_loss` measures box regression error.
- `cls_loss` measures object-class error; there is still a foreground `character` class even
  though identity recognition is excluded.
- `dfl_loss` is distribution focal loss used to refine box edges.
- `weights/best.pt` is selected by validation performance; `weights/last.pt` is simply the
  final epoch and is not automatically the better checkpoint.
- `results.png` plots train/validation losses and metrics over time.
- `val_batch*_pred.jpg` shows validation predictions produced during training.

These are **same-data fitting metrics**, not generalization metrics. Use them only to verify
that paths, labels, dependencies, and training work end to end.

### Compare predictions with ground truth

Generate confidence-bearing labels and create three overlays. Ground truth is green and
predictions are red.

```bash
uv run python -m nom_ocr_lab.run_detector predict \
  --source "$LAB_WORKSPACE/datasets/mth1000-practice/images/all" \
  --model "$LAB_WORKSPACE/models/week03-practice/weights/best.pt" \
  --name week03-practice --imgsz 640 --batch 8 --device 0

uv run python -m nom_ocr_lab.build_detection_gallery \
  --dataset "$LAB_WORKSPACE/datasets/mth1000-practice" \
  --predictions "$LAB_WORKSPACE/predictions/week03-practice/labels" \
  --output "$LAB_WORKSPACE/predictions/week03-practice-overlays" \
  --split all --selection all --limit 3
```

## 4. Controlled resolution experiment

Repeat the same 24-page training from `yolo11n.pt` with `--imgsz 1280`, the same ten epochs,
batch size, pages, and seed, but use the name `week03-practice-1280`. Evaluate it at 1280.
Compare mAP50 and mAP50–95 with the 640 run and inspect the same pages. This remains a
same-data diagnostic, not evidence of generalization.

Record whether 1280 changes missed small characters, box tightness, runtime, and GPU memory.
Use the literal terms `imgsz=640` and `imgsz=1280` in your note.

## 5. Homework: full MTH1000

Preparation creates deterministic, exhaustive, non-overlapping splits: 80% train, 10% validation, and 10% test.
Validation controls early stopping and selects `best.pt`. Inspect
the test result only after fixing the model and training settings.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data mth1000 \
  --source /data/shared/project2026/tkh-mth2k2/MTH1000 \
  --mode full

uv run python -m nom_ocr_lab.run_detector train \
  --data "$LAB_WORKSPACE/datasets/mth1000-full/data.yaml" \
  --model yolo11n.pt --name week03-mth1000 \
  --epochs 100 --imgsz 1280 --batch 8 --patience 10 --device 0

uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/mth1000-full/data.yaml" \
  --model "$LAB_WORKSPACE/models/week03-mth1000/weights/best.pt" \
  --split test --name week03-mth1000-test \
  --imgsz 1280 --batch 8 --device 0
```

Reduce `--batch` if the job runs out of GPU memory. Do not silently reduce `imgsz`.

## 6. Evaluate and predict on NomNaOCR

Use `nomnaocr/images` with the character-level `nomnaocr/detection_labels`, treated as the
course's pseudo-ground-truth reference. Do **not** use `nomnaocr/line_labels`, which describe
line regions and transcriptions. Preparation preserves nested document paths.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data nomnaocr \
  --source /data/shared/project2026/nomnaocr

uv run python -m nom_ocr_lab.run_detector evaluate \
  --data "$LAB_WORKSPACE/datasets/nomnaocr-character-detection/data.yaml" \
  --model "$LAB_WORKSPACE/models/week03-mth1000/weights/best.pt" \
  --split test --name week03-nomnaocr \
  --imgsz 1280 --batch 8 --device 0

uv run python -m nom_ocr_lab.run_detector predict \
  --source "$LAB_WORKSPACE/datasets/nomnaocr-character-detection/images/test" \
  --model "$LAB_WORKSPACE/models/week03-mth1000/weights/best.pt" \
  --name week03-nomnaocr --imgsz 1280 --batch 8 --device 0
```

Prediction labels and rendered images retain document-relative paths under
`$LAB_WORKSPACE/predictions/week03-nomnaocr/`.

## Reading the metrics

- **Precision:** among predicted character boxes, the fraction that match ground truth.
- **Recall:** among ground-truth characters, the fraction detected.
- **mAP50:** average precision when a match requires IoU at least `0.50`.
- **mAP50–95:** average precision over IoU thresholds `0.50` through `0.95`.

Week 4 derives these formulas, analyzes errors, and compares the MTH1000 checkpoint with the
Nakagawa-fine-tuned checkpoint.

## Wrap-up questions

1. Why are the practice metrics not a generalization estimate?
2. Why do we submit `best.pt` rather than automatically choosing `last.pt`?
3. What changed between the 640 and 1280 overlays, and what was the computational cost?
4. Why is a one-class detector compatible with a later recognizer containing thousands of
   character identities?

## Week 3 deliverables

1. MTH1000 practice and full `manifest.json` files with split counts.
2. The manual coordinate calculation and 640/1280 comparison note.
3. Three practice ground-truth/prediction overlays.
4. MTH1000 `best.pt` and held-out test metrics.
5. NomNaOCR metrics, nested prediction labels, and example visualizations.
6. A note distinguishing same-data fitting metrics from held-out test metrics.

Do not commit prepared datasets, weights, predictions, plots, caches, or logs.
