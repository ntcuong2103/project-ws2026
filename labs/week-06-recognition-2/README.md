# Week 6 — Recognition II: Logits and a Recognizer Forward Pass

[← Back to course README](../../README.md) · Previous: [Week 5](../week-05-recognition-1/) · Next: [Week 7](../week-07-recognition-3/)

> **Interactive Slurm reminder:** Whenever you use `srun --pty ...`, run `exit` as soon as you finish. This releases the allocation; do not leave an interactive session idle.

**Session:** each group reviews Week 5 (5 min/group), then this lab.

## Goals

- Explain a logit, token index, softmax probability, and top-*k* prediction.
- Inspect how an autoregressive recognizer's next-token logits change with its prefix.
- Run the supplied LitBTTR recognizer forward pass on Chinese, Nôm, and one detected page.

Training the recognizer is out of scope. This week does **not** use `greedy_search`, beam
search, or a student-built decoder. Those belong to Week 7.

## Starter kit and configuration

Copy [`configs/recognition.example.yaml`](../../configs/recognition.example.yaml) to a
private location and set `LAB_WORKSPACE`. It registers all checkpoint-sensitive paths,
special-token IDs, preprocessing, and the Week 4 detector handoff. The instructor provides
the compatible LitBTTR checkpoint and `nom-ids` package; neither is committed to this repo.

The Week 4 input is the complete
`$LAB_WORKSPACE/predictions/<detector-run>/labels` tree. Every source page must have a matching
label file, including an empty file when the detector found no character. Keep the relative
page path and zero-based label-row `box_id` unchanged.

## What is a logit?

At decoder step *t*, the recognizer outputs one unnormalized score for every vocabulary token.
For a batch of prefixes, the supplied adapter returns logits shaped
`[batch, step, vocabulary]`. Applying softmax converts one vocabulary row to probabilities;
`argmax` selects its largest logit. A different prefix can yield a different next-token row,
which is why decoding must feed earlier choices back to the model.

## Guided work

1. Run the hand-crafted arrays in the Week 6 notebook. Calculate argmax and softmax for the
   tiny vocabulary before executing the cells.
2. Load `vocab_ids.txt` without reordering it. Relate the five highest logits to token IDs and
   vocabulary entries.
3. Run the supplied forward adapter on the provided Chinese and Nôm crops. Record the input,
   prefix, top-*k* tokens, logits, and probabilities. These are inspections, not an accuracy
   claim for a particular font or glyph variant.
4. Run the same forward pass on one page of your Week 4 detector boxes. Save a JSONL top-token
   report containing `page_id`, `box_id`, and the inspected candidate tokens.
5. Explain why directly taking independent argmax values from a teacher-forced logit tensor is
   not the full inference decoder.

## Tasks

- [ ] Validate the tiny logits examples by hand and in the notebook.
- [ ] Inspect Chinese and Nôm crop logits with the configured recognizer.
- [ ] Produce a top-token JSONL report for one page of the team's full detector handoff.
- [ ] Document the configured checkpoint, vocabulary order, crop preprocessing, and prefix used.

## Deliverable

An executed logits notebook, Chinese/Nôm inspection notes, and one-page per-box top-token report.
Do not commit checkpoints, generated reports, or private configuration files.
