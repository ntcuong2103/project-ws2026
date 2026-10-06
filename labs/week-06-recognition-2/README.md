# Week 6 — Recognition II: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 5](../week-05-recognition-1/) · Next: [Week 7](../week-07-recognition-3/)

**Session:** each group reviews Week 5 (5 min/group), then this lab.

## Goals

- Understand encoder–decoder / attention-based recognizers (BTTR-style).
- Implement greedy decoding from per-step logits.
- Run inference on the team's real detected boxes.

## Starter kit (given)

The pretrained LitBTTR checkpoint and its forward pass (crop in, per-step logits out). Training the recognizer itself is out of scope.

## You implement

The greedy decoding loop (argmax + stopping condition), from scratch.

## Reading

Zhao et al., "BTTR," ICDAR (2021).

## Small experiment

Run your decode loop on 3–4 hand-crafted logits arrays with known expected output tokens (exact match required) before wiring it to the real model.

## Tutorial steps

1. Look at what the given forward pass returns (shape of per-step logits, vocabulary indices).
2. Implement the greedy loop: argmax per step, stop at the end token or max length.
3. Validate on the synthetic logits fixtures.
4. Run on the team's real detected boxes with the given checkpoint.
5. Save raw token sequences for the team's subset.

## Tasks

- [ ] Implement greedy decoding from raw per-step logits
- [ ] Validate on the synthetic logits fixtures (exact match required)
- [ ] Run your decode loop on the team's real detected boxes using the given checkpoint
- [ ] Save raw token sequences for the team's subset

## Deliverable

Greedy-decode implementation (validated on synthetic fixtures) + raw inference output.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
