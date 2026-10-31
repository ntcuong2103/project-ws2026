# Week 7 — Recognition III: Build the Decoder and Run the Pipeline

[← Back to course README](../../README.md) · Previous: [Week 6](../week-06-recognition-2/) · Next: [Week 8](../week-08-midterm/)

> **Interactive Slurm reminder:** Whenever you use `srun --pty ...`, run `exit` as soon as you finish. This releases the allocation; do not leave an interactive session idle.

**Session:** each group begins with a five-minute Week 6 review, then works through this lab.

You now have the pieces of the recognizer pipeline. This week, you will turn its next-token
scores into complete IDS predictions, connect those predictions back to possible Unicode
characters, and see the full page-to-recognition flow working together.

## Goals

- Build greedy autoregressive decoding from the configured recognizer's next-token logits.
- Deduplicate identical token sequences and map IDS strings back to Unicode candidates.
- Run the full page → detector boxes → recognizer → decoder pipeline and prepare the midterm.

## Decoder contract

Start every sequence with `sos_idx`. At each step, call the supplied
`next_token_logits(crops, prefixes)` primitive, choose the argmax token, and append it to the
prefix. Stop when the selected token is `eos_idx` or when you reach `max_length`. SOS and EOS
guide the decoder, so do not include them in the IDS tokens you emit. Build this loop yourself;
do not call the checkpoint’s built-in `greedy_search` or beam search.

Use the fixed fixtures to check four important edge cases exactly: a normal EOS-terminated
sequence, immediate EOS, repeated emitted tokens, and an EOS-free sequence that stops at the
maximum length.

### Tiny decoding example

Suppose `SOS = 0`, `EOS = 1`, and the ordinary IDS token IDs are `2 = ⿰`, `3 = 氵`, and
`4 = 青`. At each row, `next_token_logits(prefix)` returns one score per token in the order
`[SOS, EOS, ⿰, 氵, 青]`; choose the position with the largest score. With `max_length = 4`,
a greedy run can look like this:

> **Remember:** The prefix always starts as `[SOS]`; there is no model call with an empty
> prefix. `EOS` is not inserted by the decoder: it is a recognizer prediction. When argmax
> selects `EOS`, the decoder stops and does not emit it as part of the IDS.

| Current prefix | Logits returned by the model | Argmax token | Emitted IDS tokens |
| --- | --- | --- | --- |
| `[0]` | `[−3.0, −1.0, **4.2**, 0.3, 0.1]` | `2` (`⿰`) | `[2]` |
| `[0, 2]` | `[−2.0, 0.4, −0.5, **3.8**, 0.7]` | `3` (`氵`) | `[2, 3]` |
| `[0, 2, 3]` | `[−1.5, 0.1, 0.2, −0.3, **4.0**]` | `4` (`青`) | `[2, 3, 4]` |
| `[0, 2, 3, 4]` | `[−2.0, **5.1**, 0.3, 0.1, −0.4]` | `1` (`EOS`) | stop |

The returned token IDs are `[2, 3, 4]`: `SOS` starts the prefix and `EOS` stops decoding, but
neither is emitted. Joining the emitted tokens gives the IDS string `⿰氵青`.

## IDS composition and output contract

Join the emitted vocabulary tokens into an IDS string, then look it up in the reverse
expanded-IDS mapping. One IDS can lead to several Unicode candidates, so preserve the whole
ordered list. If there is no direct Unicode match, keep the result and record an empty candidate
list; an unmatched sequence is still useful evidence.

Write JSONL or TSV records containing `page_id`, `box_id`, normalized and pixel boxes, detector
confidence, raw token IDs, IDS string, and Unicode candidates. Use the Week 4 prediction
manifest as your input contract so every page remains accounted for, including pages with no
detector boxes.

## Tasks

- [ ] Implement and exact-match test greedy decoding on all supplied fixtures.
- [ ] Implement token-sequence deduplication and IDS→Unicode candidate lookup.
- [ ] Run the full configured pipeline on the complete Week 4 detector handoff.
- [ ] Inspect representative correct-looking, ambiguous, and unmatched IDS outputs.
- [ ] Prepare the midterm materials using [the presentation rules](../../docs/MIDTERM-PRESENTATION.md).

## Deliverable

A greedy decoder with fixture tests, full per-box recognition output, a combined detection and
recognition demo, and midterm slides. Keep predictions, weights, and private configuration out
of Git.
