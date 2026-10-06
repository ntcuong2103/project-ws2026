# Lab: Training YOLO for Nom Character Detection

## Objectives

By the end of this lab you should be able to:

1. Build a YOLO-format dataset (images + labels + `data.yaml`) from raw annotations.
2. Read and write the YOLO bounding-box annotation format by hand.
3. Explain why character *detection* is trained as **single-class**, and convert a
   multi-class label set into single-class labels.
4. Train a small YOLO model on a small character-detection dataset.
5. Visualize predicted boxes against ground truth.
6. Compute IoU by hand and interpret the mAP metrics YOLO reports.

You will use the real scripts in this repository (`prepare.py`, `train.py`, `test.py`,
`visualize.py`, `evaluate_detection_yolo.py`) and the small-run scripts in §7 — this lab is a
guided walk through the same pipeline described in `CLAUDE.md`, just on a small enough dataset
to finish in one session.

---

## 0. Setup

### 0.1 Clone the repository and link shared storage

Replace `X` below with your assigned group number. The repository lives in your group's
project directory; the workspace symlink gives you a convenient path to open in VS Code.
The shared datasets are linked once under `~/datasets`, and the repository's `datasets/`
symlink makes the paths used by the scripts resolve to that shared location.

```bash
cd
ln -s /data/projectgX workspace
ln -s /data/shared/project2026 datasets
cd workspace
git clone https://github.com/ntcuong2103/nom-ocr-project.git
cd nom-ocr-project
ln -s ~/datasets datasets
```

If any of these symlinks already exist, keep the existing link rather than replacing it.
You can verify the two data links with:

```bash
readlink -f "$HOME/datasets"
readlink -f datasets
```

The shared directory `/data/shared/project2026` contains:

| Dataset | Contents |
|---|---|
| `nakagawalab/` | 47 page images and YOLO character boxes, a class-to-character mapping for 7,504 classes, and train/validation/test manifests. This is the source dataset used by the lab. |
| `lab-char-detect/` | A single-class character-detection dataset with train/validation image and label folders plus `data.yaml`; used by the small-run scripts. |
| `nom-nakagawa-lab/` | A `labels_single/` directory for the Nom-Nakagawa lab data. |
| `nomnaocr/` | Nom OCR images, line and detection label directories, and an archive of the detection labels. |
| `tkh-mth2k2/` | The MTH1000, MTH1200, and TKH image collections, with a YAML configuration and path list; a compressed archive is also present. |

The `datasets` links point at shared storage, so generated files written under `datasets/`
are shared too. Keep temporary outputs and model runs in the project directory unless they
are meant to be shared.

### 0.2 Python environment

This lab uses [uv](https://docs.astral.sh/uv/) to manage the Python environment. Install it
if you don't have it yet (`curl -LsSf https://astral.sh/uv/install.sh | sh`), then create a
virtual environment in the repo and install the dependencies:

```bash
uv venv
uv pip install ultralytics imagesize tqdm pillow matplotlib

# Ultralytics pulls in opencv-python, which needs the system OpenGL lib (libGL.so.1).
# Swap it for the headless build so `import cv2` works on machines without it.
uv pip uninstall opencv-python
uv pip install opencv-python-headless
```

The headless build can't open GUI windows, which is fine here — every script in this lab
saves images to disk instead.

Either activate the environment (`source .venv/bin/activate`) or prefix every command in
this lab with `uv run` (e.g. `uv run python train.py`) — the two are equivalent.

Sanity-check the install:

```bash
uv run python -c "import cv2; from ultralytics import YOLO; print('ok')"
```

> **If you see `ImportError: libGL.so.1: cannot open shared object file`**, the full
> `opencv-python` package is still installed (or got reinstalled). Re-run the swap above:
> `uv pip uninstall opencv-python && uv pip install opencv-python-headless`. Installing
> anything that depends on `opencv-python` afterwards can bring it back, so repeat the swap
> if the error returns.

### 0.3 Find and select your GPU

On a GPU node, list the assigned GPU UUIDs:

```bash
nvidia-smi --query-gpu=uuid,name --format=csv,noheader
```

If the machine uses MIG, list the GPU and MIG device UUIDs with:

```bash
nvidia-smi -L
```

Use the UUID for the GPU or MIG instance assigned to you in `train_small.py` and
`predict_small.py`, replacing the example `MIG-...` value. Set
`CUDA_VISIBLE_DEVICES` before importing `ultralytics` (the scripts do this) and keep
`device=0`: CUDA exposes the selected device as device 0 inside the process. Do not copy
another user's UUID; use the device assigned to your job/session.

The lab dataset is a small slice of Nom page scans with character-level bounding boxes,
available at `~/datasets/nakagawalab` (the shared dataset link created in §0.1):

```
~/datasets/nakagawalab/
├── images/            # 47 scanned page images (~1000x771 JPEG)
├── labels/             # one YOLO .txt label file per image
├── class_mapping.txt   # per-character class id -> Unicode codepoint -> glyph
├── data.yaml
├── train.txt
├── val.txt
└── test.txt
```

Use this shared source path directly when inspecting the original dataset. The small-run
pipeline uses the prepared single-class dataset under `datasets/lab-char-detect/`.

---

## 1. The YOLO dataset format: images, labels, yaml

A YOLO detection dataset always has **three** ingredients:

```
dataset/
├── images/
│   ├── train/  img001.jpg  img002.jpg  ...
│   └── val/    img101.jpg  img102.jpg  ...
├── labels/
│   ├── train/  img001.txt  img002.txt  ...
│   └── val/    img101.txt  img102.txt  ...
└── data.yaml
```

**Rule:** every `images/.../X.jpg` must have a matching `labels/.../X.txt` with the *same
filename stem*. YOLO finds the label file by string-replacing `images` with `labels` in the
image path — get the folder names wrong and your labels silently won't load.

### 1.1 The label file format

Open one label file and look at it:

```bash
head -5 ~/datasets/nakagawalab/labels/nlvnpf-0023-022.txt
```

```
13030 0.095000 0.583009 0.030000 0.050584
1562  0.568000 0.571984 0.024000 0.031128
3976  0.128000 0.581064 0.030000 0.051881
...
```

Each line is **one object** (one character), five space-separated numbers:

```
class_id  x_center  y_center  width  height
```

- `class_id` — an integer class index (0-based).
- `x_center`, `y_center` — the box center, **normalized by image width/height** (range `0..1`).
- `width`, `height` — the box size, also normalized (range `0..1`).

Normalization is what makes YOLO labels resolution-independent: the same `.txt` file is valid
whether the image is resized to 640px or 1280px. To convert back to pixels:

```python
x_center_px = x_center * img_width
y_center_px = y_center * img_height
w_px = width * img_width
h_px = height * img_height
x1 = x_center_px - w_px / 2   # top-left corner
y1 = y_center_px - h_px / 2
```

This repo's `visualize_image_annotations()` (in `visualize.py`) does exactly this conversion
before drawing boxes — read it now, it's ~15 lines.

**Try it:** the image `nlvnpf-0023-022.jpg` is `1000x771` px. Take the first label line above
(`1562 0.568000 0.571984 0.024000 0.031128`) and compute its pixel bounding box by hand. Then
check your answer:

```python
from visualize import visualize_image_annotations
visualize_image_annotations(
    image_path="datasets/lab-char-detect/images/nlvnpf-0023-022.jpg",
    txt_path="datasets/lab-char-detect/labels/nlvnpf-0023-022.txt",
    output_path="check.jpg",
    label_map={0: "character"},
)
```

Open `check.jpg` and confirm the boxes sit tightly around individual Nom characters.

### 1.2 `data.yaml`

`data.yaml` tells YOLO where the images live and what the classes are:

```yaml
train: datasets/lab-char-detect/images/train
val: datasets/lab-char-detect/images/val
nc: 1
names: ["character"]
```

- `train` / `val` point at **image** directories (or `.txt` files listing image paths — see
  `train.txt`/`val.txt` in the shared dataset for that style). YOLO infers the label
  directories automatically by swapping `images` → `labels` in each path.
- `nc` is the number of classes, and `names` must have exactly `nc` entries, in class-id order.

---

## 2. Why single-class? (and converting multi-class → single-class)

Open `class_mapping.txt`:

```bash
head -5 ~/datasets/nakagawalab/class_mapping.txt
wc -l ~/datasets/nakagawalab/class_mapping.txt
```

```
2   78    x
3   5200  刀
4   6625  春
5   53C8  又
6   549C  咜
...
7504 lines
```

The original dataset distinguishes **7,504 different classes** — one per distinct Nom/Han
character. If you tried to train a 7,504-class detector directly on 47 pages, two problems
show up immediately:

1. **Extreme class imbalance.** A page has a few hundred character instances total, spread
   across thousands of possible classes — most classes appear 0 or 1 times in your whole
   training set. A model cannot learn a reliable appearance model for a class it has seen once.
2. **Detection and recognition are different problems.** *Detection* only needs to answer
   "is there a character here, and where exactly are its edges?" — a geometric question that
   is nearly identical for every character shape. *Recognition* ("which of 7,504 characters is
   this?") is a much harder, separate classification problem that needs far more labeled
   examples per class and is usually solved with a dedicated recognition model downstream (see
   `data_labeling.py` in this repo, which re-attaches the actual transcribed character to a
   detected box *after* detection).

So for the detection stage we **collapse every class to a single class, `0` ("character")**.
This is the `single_cls=True` flag you'll see in `train.py`. All the model has to learn is "box
around any glyph," which is a much easier, much more data-efficient task — 47 pages of a few
hundred boxes each is enough to get a usable detector, whereas it would be nowhere near enough
for 7,504-way classification.

### 2.1 Do the conversion

`prepare.py` already contains a `convert_yolo_single_class` function for this. Look at it:

```python
def convert_yolo_single_class(label_path, output_path):
    labels, x, y, w, h = list(zip(*[line.split() for line in open(label_path, 'r').readlines()]))
    with open(output_path, 'w') as f:
        for i in range(len(labels)):
            f.write(f'0 {x[i]} {y[i]} {w[i]} {h[i]}\n')
```

It keeps the box geometry and just forces every `class_id` to `0`. Run it over the whole
label directory:

```python
import glob, os
from prepare import convert_yolo_single_class

label_dir = "datasets/lab-char-detect/labels"
output_dir = "datasets/lab-char-detect/labels_single"
os.makedirs(output_dir, exist_ok=True)

for label_path in glob.glob(f"{label_dir}/*.txt"):
    output_path = label_path.replace(label_dir, output_dir)
    convert_yolo_single_class(label_path, output_path)
```

Compare a label file before and after:

```bash
head -3 datasets/lab-char-detect/labels/nlvnpf-0023-022.txt
head -3 datasets/lab-char-detect/labels_single/nlvnpf-0023-022.txt
```

The box coordinates are identical — only the leading class id changed to `0`.

---

## 3. Build the train/val split

Split the 47 images into a train and a val set (a quick 80/20 split is fine for a lab):

```python
import glob, os, random, shutil

random.seed(0)
images = sorted(glob.glob("datasets/lab-char-detect/images/*.jpg"))
random.shuffle(images)

n_val = max(1, int(0.2 * len(images)))
splits = {"val": images[:n_val], "train": images[n_val:]}

for split, split_images in splits.items():
    img_out = f"datasets/lab-char-detect/images/{split}"
    lbl_out = f"datasets/lab-char-detect/labels/{split}"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img_path in split_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        shutil.copy(img_path, f"{img_out}/{stem}.jpg")
        shutil.copy(f"datasets/lab-char-detect/labels_single/{stem}.txt", f"{lbl_out}/{stem}.txt")

print({k: len(v) for k, v in splits.items()})
```

Now write `datasets/lab-char-detect/data.yaml`:

```yaml
train: datasets/lab-char-detect/images/train
val: datasets/lab-char-detect/images/val
nc: 1
names: ["character"]
```

---

## 4. Train a small model

Compare against this repo's `train.py`, then run a lab-sized version — small model
(`yolo11n`), small image size, few epochs, so it finishes on CPU or a single GPU within the
lab session:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")       # pretrained COCO checkpoint, used as a starting point
model.train(
    data="datasets/lab-char-detect/data.yaml",
    epochs=30,
    imgsz=640,
    single_cls=True,             # collapse to one class even if labels weren't pre-converted
    batch=8,
    patience=10,
    project="lab-runs",
    name="character-detect",
)
```

Notes on the flags, and how they connect to what you just learned:

- `single_cls=True` — reinforces §2: even though we already rewrote the labels to class `0`,
  this flag also tells YOLO to ignore whatever class id is in the file and treat every box as
  the same class. In production (`train.py`) both are typically used together for safety.
  Skipping the manual conversion in §2 and only setting `single_cls=True` also works — it's
  worth trying both ways and confirming the results are the same, to understand what the flag
  actually does.
- `imgsz=640` — YOLO resizes/pads every image to a square of this size before feeding it to
  the network. The production `train.py` uses `imgsz=1280` because these are full page scans
  with small, dense characters — 640 is used here only to keep the lab fast; note in your
  results whether small characters get missed more at the lower resolution.
- `epochs` / `patience` — `patience` stops training early if validation performance hasn't
  improved for that many epochs (a lab-friendly time-saver).

While it trains, note the printed columns: `box_loss`, `cls_loss`, `dfl_loss` for both `train`
and `val`, plus the running `mAP50` and `mAP50-95` on the validation set after every epoch —
you'll interpret these numbers in §6.

With the default Ultralytics run directory, training writes everything under
`runs/detect/lab-runs/character-detect/`:

```
runs/detect/lab-runs/character-detect/
├── weights/best.pt        # best checkpoint by validation metric
├── weights/last.pt        # checkpoint from the final epoch
├── results.png            # loss & metric curves over training
├── confusion_matrix.png
└── val_batch0_pred.jpg    # example predictions on a validation batch
```

Open `results.png` and `val_batch0_pred.jpg` now — this is your first look at whether the
model is learning to find character boxes at all before you dig into the numbers.

---

## 5. Visualize predictions

### 5.1 Quick built-in visualization

```python
from ultralytics import YOLO

model = YOLO("runs/detect/lab-runs/character-detect/weights/best.pt")
model.predict(
    "datasets/lab-char-detect/images/val",
    save=True,
    conf=0.25,
    project="lab-runs",
    name="predict",
)
```

This writes annotated images to `runs/detect/lab-runs/predict/`. Open a few and compare them
side-by-side with the ground-truth boxes in `datasets/lab-char-detect/images/val/*.jpg` +
`datasets/lab-char-detect/labels/val/*.txt`.

### 5.2 Side-by-side with this repo's `visualize.py`

To directly compare predicted boxes against ground truth using the same drawing code from §1:

```python
model.predict(
    "datasets/lab-char-detect/images/val",
    save_txt=True, save_conf=True, conf=0.25,
    project="lab-runs", name="predict_labels",
)
```

```python
from visualize import visualize_image_annotations

img = "datasets/lab-char-detect/images/val/nlvnpf-0991-01-025.jpg"
visualize_image_annotations(img, "datasets/lab-char-detect/labels/val/nlvnpf-0991-01-025.txt",
                             "gt.jpg", {0: "character"})
visualize_image_annotations(img, "runs/detect/lab-runs/predict_labels/labels/nlvnpf-0991-01-025.txt",
                             "pred.jpg", {0: "character"})
```

Look for three kinds of mistakes: **missed characters** (a ground-truth box with nothing
predicted near it — a false negative), **false detections** (a predicted box with no matching
ground truth — a false positive), and **loose/tight boxes** (a predicted box roughly on a
character but with poorly fitted edges). You'll turn these observations into numbers next.

---

## 6. Understanding IoU and mAP

### 6.1 Intersection over Union (IoU)

IoU measures how well two boxes overlap — it's the basis for deciding whether a prediction
"counts" as matching a ground-truth box.

```
        ┌────────────┐
        │  ground    │
        │  truth  ┌──┼─────────┐
        │         │██│ overlap │
        └─────────┼──┘         │
                   │ prediction│
                   └───────────┘

IoU = area(overlap) / area(union)
    = area(GT ∩ Pred) / (area(GT) + area(Pred) − area(GT ∩ Pred))
```

IoU is always between `0` (no overlap) and `1` (identical boxes).

**By hand:** take two boxes and compute IoU yourself.

```python
def iou(box_a, box_b):
    # boxes as (x1, y1, x2, y2) in pixels
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

print(iou((100, 100, 150, 150), (120, 120, 170, 170)))  # partial overlap
print(iou((100, 100, 150, 150), (100, 100, 150, 150)))  # identical -> 1.0
print(iou((100, 100, 150, 150), (200, 200, 250, 250)))  # no overlap -> 0.0
```

This is a from-scratch version of the same math `evaluate_detection_yolo.py` delegates to
Ultralytics' `DetectionValidator._process_batch` — worth reading that repo file end-to-end now
that you know what it's computing under the hood.

**IoU threshold:** a detector's output is only "correct" for a given ground-truth box if IoU
is above some threshold — commonly `0.5`. If a prediction overlaps a ground-truth character by
IoU `0.3`, it's counted as a miss even though it's roughly in the right place.

### 6.2 From IoU to precision/recall to mAP

For a chosen IoU threshold (say 0.5), every prediction is bucketed as:

- **True Positive (TP)** — a predicted box that matches a ground-truth box with IoU ≥
  threshold (and each ground-truth box can only be matched once).
- **False Positive (FP)** — a predicted box with no matching ground-truth box.
- **False Negative (FN)** — a ground-truth box with no matching prediction.

From these counts:

```
Precision = TP / (TP + FP)     # of the boxes I predicted, how many were real?
Recall    = TP / (TP + FN)     # of the real boxes, how many did I find?
```

There's a trade-off: lowering the confidence threshold (`conf=` in `model.predict`) finds more
true characters (higher recall) but also produces more spurious boxes (lower precision).
Plotting precision against recall as you sweep the confidence threshold gives a
**precision-recall curve**; the area under that curve for one class is its **Average
Precision (AP)**. Since our dataset (after §2) has one class, our `mAP` — "mean" AP averaged
over classes — is just that one class's AP.

Ultralytics reports two AP numbers, both in the training log and in `evaluate_detection_yolo.py`'s
output dict:

- **mAP50** — AP computed with the IoU-match threshold fixed at 0.5 (a lenient "roughly in the
  right place" criterion).
- **mAP50-95** — AP averaged over ten IoU thresholds, `0.5, 0.55, ..., 0.95` (the stricter
  COCO-style metric). This rewards boxes that are not just present but tightly fitted — a
  detector can have a good mAP50 but a much lower mAP50-95 if its boxes are consistently a bit
  loose.

### 6.3 Run the repo's evaluator

```bash
uv run python evaluate_detection_yolo.py \
  --gt_dir datasets/lab-char-detect/labels/val \
  --pred_dir runs/detect/lab-runs/predict_labels/labels \
  --iou_thres 0.5
```

This prints a `results_dict` with keys like `metrics/precision(B)`, `metrics/recall(B)`,
`metrics/mAP50(B)`, `metrics/mAP50-95(B)`. Compare these numbers to what `model.val()` reports
directly:

```python
metrics = model.val(data="datasets/lab-char-detect/data.yaml")
print(metrics.box.map50, metrics.box.map)   # mAP50, mAP50-95
```

They should agree closely (small differences can come from `evaluate_detection_yolo.py`
matching files by filename stem rather than using YOLO's internal dataloader).

---

## 7. Small-run Python scripts

These are the complete small-run scripts from the repository. Run them from the repository
root with the environment created in §0.2. The scripts read/write the prepared
`datasets/lab-char-detect/` tree (linked to shared storage by the setup above).

### 7.1 Convert labels to one class — `convert_to_single_class.py`

This preserves each box and changes only its class id to `0`:

```python
import glob, os
from prepare import convert_yolo_single_class

label_dir = "datasets/lab-char-detect/labels"
output_dir = "datasets/lab-char-detect/labels_single"
os.makedirs(output_dir, exist_ok=True)

for label_path in glob.glob(f"{label_dir}/*.txt"):
    output_path = label_path.replace(label_dir, output_dir)
    convert_yolo_single_class(label_path, output_path)
```

### 7.2 Split images into train and validation — `split_dataset.py`

```python
import glob, os, random, shutil

random.seed(0)
images = sorted(glob.glob("datasets/lab-char-detect/images/*.jpg"))
random.shuffle(images)

n_val = max(1, int(0.2 * len(images)))
splits = {"val": images[:n_val], "train": images[n_val:]}

for split, split_images in splits.items():
    img_out = f"datasets/lab-char-detect/images/{split}"
    lbl_out = f"datasets/lab-char-detect/labels/{split}"
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    for img_path in split_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        shutil.copy(img_path, f"{img_out}/{stem}.jpg")
        shutil.copy(f"datasets/lab-char-detect/labels_single/{stem}.txt", f"{lbl_out}/{stem}.txt")

print({k: len(v) for k, v in splits.items()})
```

### 7.3 Train — `train_small.py`

Replace the GPU/MIG UUID with the UUID found in §0.3 before running.

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"  # specify which GPU to use

from ultralytics import YOLO

model = YOLO("yolo11n.pt")       # pretrained COCO checkpoint, used as a starting point
model.train(
    data="datasets/lab-char-detect/data.yaml",
    epochs=20,
    imgsz=640,
    single_cls=True,             # collapse to one class even if labels weren't pre-converted
    batch=8,
    patience=10,
    project="lab-runs",
    name="character-detect",
    device=0
)
```

### 7.4 Visualize an annotated image — `run_visualize.py`

```python
from visualize import visualize_image_annotations
visualize_image_annotations(
    image_path="datasets/lab-char-detect/images/nlvnpf-0023-022.jpg",
    txt_path="datasets/lab-char-detect/labels/nlvnpf-0023-022.txt",
    output_path="check.jpg",
    label_map={0: "character"},
)
```

### 7.5 Predict — `predict_small.py`

Replace the GPU/MIG UUID with the UUID found in §0.3 before running.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"  # specify which GPU to use

from ultralytics import YOLO

model = YOLO("runs/detect/lab-runs/character-detect/weights/best.pt")
model.predict(
    "datasets/lab-char-detect/images/train",
    save=True,
    conf=0.1,
    project="lab-runs",
    name="predict",
    imgsz=640,
    device=0
)
```

---

## 8. Wrap-up questions

Answer these from your own run, not from the text above:

1. Look at 3 false negatives (missed characters) in your visualizations. Do they share a
   common trait (very small, touching another character, low contrast, near the page edge)?
2. Retrain with `imgsz=1280` instead of `640` (the setting `train.py` actually uses for full
   pages). Does mAP50-95 improve more than mAP50? What does that tell you about *why* the
   production script uses a larger image size for this data?
3. Your `data.yaml` says `nc: 1`. What would happen — concretely, to the loss and to the
   printed metrics — if you set `nc: 7504` and used the original multi-class labels from
   `class_mapping.txt` instead of the single-class ones? You don't have to fully retrain this;
   reason about it from what you now know about precision/recall per class.
4. Pick one prediction with IoU between 0.3 and 0.5 against its nearest ground-truth box.
   Would it count as correct at the `mAP50` threshold? At `mAP50-95`? Explain using the IoU
   formula from §6.1.
