# Week 4 — Fine-Tuning on Nôm and External Evaluation

[← Week 3](../week-03-detection-1/) · [Back to course README](../../README.md) · [Week 5 →](../week-05-recognition-1/)

This guided lab adapts the Week 3 character detector to a small Nôm collection. You will use
the supplied Nakagawa splits and compare the detector before and after adaptation on an external
NomNaOCR reference set. This is an introduction to the workflow, not a complete reproduction of
a research experiment.

## Outcomes

By the end of the lab, your team can:

1. explain why thousands of character identities become one detection class;
2. fine-tune from a previous checkpoint without leaking validation or test data;
3. apply the Week 3 metric interpretation when reading held-out results; and
4. make a qualitative before/after comparison on the same external data.

## 1. Verify the Week 3 handoff

```bash
export LAB_WORKSPACE=/data/projectgX/lab-exercise
test -f "$LAB_WORKSPACE/models/week03-mth1200/weights/best.pt"
test -f "$LAB_WORKSPACE/models/week03-mth1200/weights/last.pt"
mkdir -p "$LAB_WORKSPACE/logs"
```

Do not substitute a checkpoint selected using MTH1000 test results.

## 2. Build your Nakagawa dataset workspace

Nakagawa contains many character identities, but the detector only learns the shared geometric
task “where is a character?” Recognition models later learn “which character is it?” You will
create the dataset view yourself. The source is read-only; create everything below in your own
lab workspace, not under `/data/shared/project2026`.

```bash
export NAKAGAWA_SOURCE=/data/shared/project2026/nakagawalab
export NAKAGAWA_DATASET="$LAB_WORKSPACE/datasets/nakagawa-character-detection"
mkdir -p "$NAKAGAWA_DATASET"/{images,labels}
```

The supplied `train.txt`, `val.txt`, and `test.txt` files are the authoritative split manifests.
Each row is an image path, such as
`yolo_data/images/nlvnpf-0991-01-047.jpg`. Preserve the listed split; do not shuffle, resplit,
or move examples between the files. The manifests use Windows-style separators, so convert `\` to
`/` before treating a row as a path.

For every non-empty row in each manifest:

1. Copy the image from `$NAKAGAWA_SOURCE/yolo_data/images/<relative-image>` to
   `$NAKAGAWA_DATASET/images/<split>/<relative-image>`.
2. Copy its matching label from
   `$NAKAGAWA_SOURCE/yolo_data/labels/<relative-image-with-.txt-extension>` to
   `$NAKAGAWA_DATASET/labels/<split>/<relative-image-with-.txt-extension>`.
3. Rewrite the first field of every non-empty YOLO label row to class `0`. Keep its four
   normalized box coordinates unchanged.

For example, Nakagawa's raw multi-identity row

```text
13030 0.095000 0.583009 0.030000 0.050584
```

must become the one-class detection row

```text
0 0.095000 0.583009 0.030000 0.050584
```

The original first value is a character identity for recognition, not a detection class. Do not
use it for this detector.

Implement this process yourselves. Your `data.yaml` must identify the dataset root, point its
training, validation, and test entries at the respective `images/` directories, and declare one
class named `character` with ID `0`.

Inspect three source rows and their dataset counterparts. Confirm that the first field changes to
`0`, the four coordinates do not change, every image has a matching label, and no page appears in
more than one split. Include a short `README.md` or `manifest.md` in the dataset directory
recording the source path, the three split counts, and the command you used to build it. Do not
commit this generated dataset.

## 3. Fine-tune through Slurm

Write and submit your own Week 4 `train.sbatch`. It must start from the Week 3 `best.pt`, train
only on Nakagawa train, use Nakagawa validation for early stopping and checkpoint selection, and
write all artifacts to your group workspace. Include the job resources, exact command, and any
non-default hyperparameters in your submission.

Inspect the Slurm log, `results.csv`, plots, `weights/best.pt`, and `weights/last.pt`. The final
Nakagawa test set is for evaluation, not model selection.

## 4. Prepare the external NomNaOCR reference set

The server provides the smaller Roboflow export as a ZIP. The helper copies every labeled page
from all archive partitions into one external test view. The original archive partition remains
in each relative path so equal filenames cannot collide.

```bash
uv run python -m nom_ocr_lab.prepare_detection_data prepare-nomnaocr-archive \
  --archive /data/shared/project2026/nom-detection-val.v2i.yolov11.zip
```

Inspect `datasets/nomnaocr-character-detection/manifest.json` and `data.yaml`. Do not use this
external set for early stopping or hyperparameter selection.

## 5. Evaluate before and after

Write and submit your own Week 4 `evaluate.sbatch`. It must evaluate the Nakagawa checkpoint on
Nakagawa test, evaluate the Week 3 and Week 4 checkpoints on identical NomNaOCR data, and create
the `predictions/week04-nomnaocr/` handoff used by later recognition labs. Use the Week 3 metric
definitions when interpreting the results; neither Nakagawa test nor NomNaOCR may drive model
selection.

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
2. Your Week 4 training and evaluation `sbatch` files, Slurm logs, curves, `best.pt`, and `last.pt`.
3. Nakagawa held-out test metrics.
4. Week 3 versus Week 4 NomNaOCR metrics and a short qualitative interpretation.
5. The complete `predictions/week04-nomnaocr/` handoff and several inspected error examples.

Grading emphasizes correct workflow, artifact organization, metric understanding, and thoughtful
interpretation—not reproducing the reference values. Do not commit generated datasets, weights,
logs, metrics, predictions, or plots.
