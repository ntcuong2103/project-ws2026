# Week 13 — Pseudo-labeling II & Metrics: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 12](../week-12-pseudo-labeling-1/) · Next: [Week 14](../week-14-integration/)

**Session:** each group reviews Week 12 (5 min/group), then this lab.

## Goals

- Implement the modified CER and confidence-band precision/recall.
- Build a small human-verified holdout and evaluate on it.

## Starter kit (given)

The annotation review app, as-is. Building a web annotation tool is out of scope.

## You implement

- The modified CER metric (§3.4): expand each character to its Unicode-variant set before computing edit distance.
- The confidence-band precision/recall computation.

## Reading

Source paper §3.3–3.4, §4.2–4.4.3 (re-read).

## Small experiment

- Hand-compute the modified CER on 2–3 short strings containing at least one known Unicode-variant pair.
- Hand-verify precision/recall on a 10-item toy confusion table.

Check both against your implementation before touching the real holdout.

## Tutorial steps

1. Work the toy examples by hand.
2. Implement modified CER and verify.
3. Implement confidence-band precision/recall and verify on the 10-item table.
4. Use the annotation app to build a small human-verified holdout for your subset.
5. Compute both metrics on the holdout; run 1–2 more self-training iterations if time allows.

## Tasks

- [ ] Implement modified CER; verify against your hand-computed toy examples
- [ ] Implement precision/recall by confidence band; verify on the 10-item toy table
- [ ] Use the annotation app to build a small human-verified holdout for your subset
- [ ] Compute both metrics on the real holdout; run 1–2 more self-training iterations if time allows

## Deliverable

Modified CER + confidence-band precision/recall, validated on toy examples, then run against the team's holdout.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
