# Week 6 — Recognition II: Logits and a Recognizer Forward Pass

[← Back to course README](../../README.md) · Previous: [Week 5](../week-05-recognition-1/) · Next: [Week 7](../week-07-recognition-3/)

> **Interactive Slurm reminder:** Whenever you use `srun --pty ...`, run `exit` as soon as you finish. This releases the allocation; do not leave an interactive session idle.

**Session:** each group begins with a five-minute Week 5 review, then works through this lab.

This week is about looking inside the recognizer before asking it to decode a character on its
own. You will trace how it scores possible next tokens and connect those scores to the crops
and IDS vocabulary you prepared last week.

## Goals

- Explain a logit, token index, softmax probability, and top-*k* prediction.
- Inspect how an autoregressive recognizer's next-token logits change with its prefix.
- Run the supplied LitBTTR recognizer forward pass on Chinese, Nôm, and one detected page.

## Starter kit and configuration

Copy [`configs/recognition.example.yaml`](../../configs/recognition.example.yaml) to a private
location and set `LAB_WORKSPACE` plus `COURSE_REPOSITORY` (the absolute path to this course
checkout). The configuration keeps the checkpoint-sensitive paths, special-token IDs,
preprocessing settings, bundled `nom-ids` resources, and Week 4 detector handoff together.

### Copy the provided recognizer checkpoint

The shared checkpoint is read-only. Before using it, make one copy in your team’s persistent
data workspace. This keeps any write-capable tool away from the shared original. From your team
workspace, run:

```bash
mkdir -p /data/projectgX/models/recognizer
cp --preserve=mode,timestamps \
  /data/shared/project2026/checkpoint-sample-nom-ids/epoch=199-step=19048-val_ExpRate=0.9508.ckpt \
  /data/projectgX/models/recognizer/litbttr.ckpt
sha256sum \
  /data/shared/project2026/checkpoint-sample-nom-ids/epoch=199-step=19048-val_ExpRate=0.9508.ckpt \
  /data/projectgX/models/recognizer/litbttr.ckpt
```

Replace `projectgX` with your assigned group workspace. Check that the two hashes match, then
keep the copy under `/data/projectgX` rather than in Git. Finally, point
`paths.recognizer_checkpoint` in your private configuration to
`/data/projectgX/models/recognizer/litbttr.ckpt`.

Your Week 4 handoff is the complete
`$LAB_WORKSPACE/predictions/<detector-run>/labels` tree. Make sure every source page has a
matching label file, even when the detector found no characters. Keep both the relative page
path and the zero-based label-row `box_id` unchanged so later stages can trace each crop back.

## Crop-output contract

Implement the crop stage in your team repository. For every detector row, create one RGB crop
and one manifest record. Preserve the zero-based detector-row `box_id`, class ID, optional
confidence, normalized YOLO box, pixel box, expanded crop box, and crop path. At this point the
crops are still unlabeled—do not assign character identities.

Center each crop on its detector box and use a nominal square side of
`1.2 × max(pixel_width, pixel_height)`, clipped to the page boundary. A crop at an edge may
therefore end up smaller or non-square. Choose one consistent conversion from normalized to
integer pixels, record the coordinates actually used, and name the manifest records and crops
so each one can be traced back to its source page and `box_id`.

## What is a logit?

At decoder step *t*, the recognizer gives every vocabulary token an unnormalized score. That
score is a *logit*. For a batch of prefixes, the supplied adapter returns logits shaped
`[batch, step, vocabulary]`. Softmax turns one vocabulary row into probabilities, while
`argmax` chooses the largest logit. Because a different prefix can change the next-token row,
a decoder must feed earlier choices back into the model.

## Guided work

1. Start with the hand-crafted arrays in the Week 6 notebook. Work out the argmax and softmax
   for the tiny vocabulary on paper before you run the cells.
2. Load `vocab_ids.txt` without reordering it, then connect the five largest logits to their
   token IDs and vocabulary entries.
3. Use the supplied forward adapter on the provided Chinese and Nôm crops. Record the input,
   prefix, top-*k* tokens, logits, and probabilities. Treat these as a close look at model
   behaviour, not an accuracy claim for one font or glyph variant.
4. Repeat the forward pass for one page of your Week 4 detector boxes. Save a JSONL top-token
   report with `page_id`, `box_id`, and the candidate tokens you inspected.
5. In your own words, explain why independently taking argmax values from a teacher-forced
   logit tensor is not yet a complete inference decoder.

## Tasks

- [ ] Validate the tiny logits examples by hand and in the notebook.
- [ ] Inspect Chinese and Nôm crop logits with the configured recognizer.
- [ ] Produce a top-token JSONL report for one page of the team's full detector handoff.
- [ ] Document the configured checkpoint, vocabulary order, crop preprocessing, and prefix used.

## Deliverable

An executed logits notebook, Chinese/Nôm inspection notes, and a one-page, per-box top-token
report. Keep checkpoints, generated reports, and private configuration files out of Git.
