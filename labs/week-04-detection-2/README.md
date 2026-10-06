# Week 4 — Detection II: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 3](../week-03-detection-1/) · Next: [Week 5](../week-05-recognition-1/)

**Session:** each group reviews Week 3 (5 min/group), then this lab.

## Goals

- Fine-tune the MTHv2-pretrained detector on the TUAT-Nakagawa (NakagawaLab) dataset.
- Measure Precision/Recall/mAP@50.
- Classify detector error modes: merges, splits, spurious and missed boxes.

## Starter kit (given)

Precision/Recall/mAP@50 evaluation using ultralytics.

## You implement

Fine-tuning of the pretrained MTHv2 detector on NakagawaLab, then check the metrics after training.

## Tutorial steps

1. Prepare the NakagawaLab data in YOLO format.
2. Fine-tune from the Week 3 MTHv2 checkpoint.
3. Run ultralytics' evaluation and record P / R / mAP@50.
4. Visualize detections on 5–10 pages with the Week 3 script.
5. Tag at least 10 errors by type and write the error-analysis note.

## Tasks

- [ ] Fine-tune the MTHv2-pretrained YOLO detector on the TUAT-Nakagawa (NakagawaLab) dataset
- [ ] Compute Precision/Recall/mAP@50 with ultralytics' built-in evaluation after training
- [ ] Visualize detections on 5–10 pages using your Week 3 visualization script
- [ ] Tag at least 10 detector errors by type (merge/split/spurious/missed); write a half-page error-analysis note

## Deliverable

Fine-tuned detector + ultralytics metrics report + error-analysis gallery.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
