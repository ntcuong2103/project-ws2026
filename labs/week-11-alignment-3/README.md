# Week 11 — Alignment III: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 10](../week-10-alignment-2/) · Next: [Week 12](../week-12-pseudo-labeling-1/)

**Session:** each group reviews Week 10 (5 min/group), then this lab.

## Goals

- Add the graded confidence score to the alignment.
- Implement the binary-flag baseline for comparison.
- Run both on the team's full subset.

## Starter kit (given)

None new.

## You implement

- The confidence score s_i = 1 − ED(p_i, g_j) / max(|p_i|, |g_j|) (Eq. 2) on top of your Week 10 alignment.
- A binary-flag variant (accept only on exact match) for the Table 2 comparison.

## Small experiment

Compute s_i by hand for 3–4 aligned pairs in your Week 10 toy example and check your code's output matches.

## Tutorial steps

1. Hand-compute s_i for 3–4 pairs from the Week 10 toy example.
2. Add the confidence computation to the alignment output.
3. Implement the binary-flag variant.
4. Run both on the full assigned subset and save both label sets.
5. Plot the distribution of s_i across the subset.

## Tasks

- [ ] Add confidence-score computation to your alignment; verify by hand on the toy example
- [ ] Implement the binary-flag variant of the same alignment
- [ ] Run both variants on the team's full assigned subset; save both label sets
- [ ] Plot the distribution of s_i values across the team's subset

## Deliverable

Confidence-weighted alignment + binary-flag baseline, both run on the team's subset.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
