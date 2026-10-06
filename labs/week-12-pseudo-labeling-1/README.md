# Week 12 — Pseudo-labeling I: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 11](../week-11-alignment-3/) · Next: [Week 13](../week-13-pseudo-labeling-2/)

**Session:** each group reviews Week 11 (5 min/group), then this lab.

## Goals

- Design the confidence-weighted training loop.
- Run one real recognize → align → weight → fine-tune iteration.

## Starter kit (given)

A minimal fine-tuning loop skeleton (data loading + checkpointing only). The loss function is a stub.

## You implement

- A confidence-weighted loss: `loss = sum(s_i * per_box_loss_i)`.
- The retrain / re-infer / re-align iteration loop.

## Small experiment

On a 5-example synthetic mini-batch with hand-picked s_i values, verify that your weighted loss gives the number you computed by hand, before running real fine-tuning.

## Tutorial steps

1. Compute the expected weighted loss for the mini-batch by hand.
2. Implement the loss and check it matches.
3. Wire it into the fine-tuning skeleton.
4. Run one iteration on the team's subset.
5. Log CER and labeled coverage before and after.

## Tasks

- [ ] Implement the confidence-weighted loss; verify on the synthetic mini-batch
- [ ] Wire it into a fine-tuning loop (recognize → align → weight → fine-tune)
- [ ] Run one real iteration on the team's subset
- [ ] Log CER and labeled coverage before/after

## Deliverable

Confidence-weighted fine-tuning loop, validated on a toy mini-batch, run for one real iteration.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
