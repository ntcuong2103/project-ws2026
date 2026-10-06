# Week 3 — Detection I: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 2](../week-02-foundations/) · Next: [Week 4](../week-04-detection-2/)

**Session:** each group reviews Week 2 (5 min/group), then this lab.

## Goals

- Understand the YOLOv11n architecture and annotation format.
- Train a character detector with transfer learning from MTHv2, first on a small sample and then at full scale.
- Run detection on the team's NomNaOCR subset and inspect the results.

## Starter kit

YOLO training script, a small number of MTHv2 samples, YOLO detection script, visualization script.



## You implement

The YOLO training run on the full MTHv2 samples, then detection and visualization on NomNaOCR.

## Reading

Khanam & Hussain, "YOLOv11: An Overview of the Key Architectural Enhancements" (2024); Ma et al., ICFHR (2020).

## Tutorial steps

1. Look at the YOLO label format (class, normalized box) on a few MTHv2 samples.
2. Run the given training script on the small MTHv2 sample; confirm it runs end-to-end.
3. Scale training to the full MTHv2 set on the lab GPUs.
4. Run the trained detector on the team's NomNaOCR subset.
5. Visualize detections with the given script and review a few pages.

## Tasks

- [ ] Run the given YOLO training script on the small MTHv2 sample; confirm the training loop runs end-to-end
- [ ] Scale training up to the full set of MTHv2 samples
- [ ] Run the trained detector on the team's assigned NomnaOCR subset
- [ ] Visualize detections with the given visualization script and review a few pages

## Deliverable

Trained YOLO detector (validated first on the small MTHv2 sample, then trained at full scale) + detection output and visualizations on the team's NomNaOCR subset.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
