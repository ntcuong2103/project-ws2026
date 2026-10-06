# Week 14 — Integration: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 13](../week-13-pseudo-labeling-2/) · Next: [Week 15](../week-15-final/)

**Session:** each group reviews Week 13 (5 min/group), then this lab.

## Goals

- Test robustness to detector segmentation noise.
- Produce the per-document-type comparison (binary-flag vs. graded alignment).
- Finalize report materials and rehearse the final presentation.

## Starter kit (given)

None.

## You implement

A noise-injection function that duplicates or drops boxes at a controlled rate.

## Reading

Source paper §4.4.5 and §5 (Discussion) (re-read).

## Small experiment

On a synthetic list of 10 boxes, inject 30% noise. Verify that the resulting count and the affected boxes match what you expect, before running at the paper's 5% / 10% rates on real data.

## Tutorial steps

1. Decide what you expect from the 10-box example (counts, which boxes).
2. Implement duplicate/drop injection and verify.
3. Run at 5% and 10% on the team's real detection output.
4. Build the per-document-type table for the team's folder.
5. Finalize report tables/figures; do a full dry-run of the final presentation.

## Tasks

- [ ] Implement box duplication/drop noise injection; verify on the 10-box synthetic list
- [ ] Run the noise experiment at 5% and 10% on the team's real detection output
- [ ] Build the per-document-type table (binary-flag vs. graded alignment) for the team's folder
- [ ] Finalize report tables/figures; full dry-run of the final presentation

## Deliverable

Noise-injection implementation (validated on a toy list) + robustness results + finalized report materials.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
