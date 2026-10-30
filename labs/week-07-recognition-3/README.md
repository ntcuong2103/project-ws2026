# Week 7 — Recognition III: Build the Decoder and Run the Pipeline

[← Back to course README](../../README.md) · Previous: [Week 6](../week-06-recognition-2/) · Next: [Week 8](../week-08-midterm/)

> **Interactive Slurm reminder:** Whenever you use `srun --pty ...`, run `exit` as soon as you finish. This releases the allocation; do not leave an interactive session idle.

**Session:** each group reviews Week 6 (5 min/group), then this lab.

## Goals

- Build greedy autoregressive decoding from the configured recognizer's next-token logits.
- Deduplicate identical token sequences and map IDS strings back to Unicode candidates.
- Run the full page → detector boxes → recognizer → decoder pipeline and prepare the midterm.

## Decoder contract

Start every sequence with `sos_idx`. At each step, call the supplied
`next_token_logits(crops, prefixes)` primitive, choose the argmax token, append it to the
prefix, and stop when the selected token is `eos_idx` or `max_length` is reached. Do not emit
SOS or EOS as IDS tokens. Do not call the checkpoint's built-in `greedy_search` or beam search.

Use the fixed fixtures to prove these cases exactly: a normal EOS-terminated sequence, immediate EOS,
repeated emitted tokens, and an EOS-free sequence stopped by the maximum length.

## IDS composition and output contract

Join emitted vocabulary tokens into an IDS string, then use the reverse expanded-IDS mapping.
One IDS can map to multiple Unicode candidates; preserve the entire ordered candidate list.
An IDS with no direct Unicode match must be reported with an empty candidate list, not discarded.

Write JSONL or TSV records with `page_id`, `box_id`, normalized and pixel boxes, detector
confidence, raw token IDs, IDS string, and Unicode candidates. The Week 4 prediction manifest
is the input contract, so every page is accounted for even if it has no detector boxes.

## Tasks

- [ ] Implement and exact-match test greedy decoding on all supplied fixtures.
- [ ] Implement token-sequence deduplication and IDS→Unicode candidate lookup.
- [ ] Run the full configured pipeline on the complete Week 4 detector handoff.
- [ ] Inspect representative correct-looking, ambiguous, and unmatched IDS outputs.
- [ ] Prepare the midterm materials using [the presentation rules](../../docs/MIDTERM-PRESENTATION.md).

## Deliverable

Greedy decoder with fixture tests, full per-box recognition output, a combined detection and
recognition demo, and midterm slides. Do not commit predictions, weights, or private configs.
