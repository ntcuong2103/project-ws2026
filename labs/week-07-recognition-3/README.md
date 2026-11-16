# Week 7 — Recognition III: Build the Bidirectional Decoder and Run the Pipeline

[← Back to course README](../../README.md) · Previous: [Week 6](../week-06-recognition-2/) · Next: [Week 8](../week-08-midterm/)

> **Interactive Slurm reminder:** Whenever you use `srun --pty ...`, run `exit` as soon as you finish. This releases the allocation; do not leave an interactive session idle.

**Session:** each group begins with a five-minute Week 6 review, then works through this lab.

You now have the pieces of the recognizer pipeline. This week, you will turn its next-token
scores into complete IDS predictions, connect those predictions back to possible Unicode
characters, and see the full page-to-recognition flow working together.

The recognizer is BTTR: one decoder trained to read every IDS sequence both left-to-right and
right-to-left. The course decoder uses both directions, exactly as Dr. Cuong's reference
notebook does.

## Goals

- Build greedy autoregressive decoding from the configured recognizer's next-token logits.
- Extend it to bidirectional decoding: decode in both directions, re-score each result with the
  opposite direction, and keep the better one.
- Deduplicate identical token sequences and map IDS strings back to Unicode candidates.
- Run the full page → detector boxes → recognizer → decoder pipeline and prepare the midterm.

## Reference notebook

Dr. Cuong's walkthrough is the ground truth for how the decoder behaves:
[`nom-ids@project-2026/evaluate.ipynb`](https://github.com/ntcuong2103/nom-ids/blob/project-2026/evaluate.ipynb).
It loads the provided checkpoint, decodes a batch of crops in both directions, prints the
step-by-step tables for one crop, and compares greedy with beam search. Read sections 2.1, 3 and
4 before you write code.

To run it yourself, use the `nom-ids` clone from Week 6 and this repository's `recognition`
environment:

```bash
cd "$LAB_WORKSPACE/models/nom-ids"
ln -sfn /data/shared/project2026 datasets
srun --gres=gpu:a100_2g.10gb:1 -c 4 --mem=16G -t 01:00:00 --pty bash
uv run --project /home/projectgX/project-ws2026 --group recognition \
  jupyter lab evaluate.ipynb
```

Delete the line that sets `os.environ["CUDA_VISIBLE_DEVICES"]` in the first code cell. Slurm
already gives your job its GPU, and the hard-coded device ID belongs to someone else's session.

## Step 1: one direction (left-to-right)

Start every sequence with `sos_idx`. At each step, call the supplied
`next_token_logits(crops, prefixes)` primitive, choose the argmax token, and append it to the
prefix. Stop when the selected token is `eos_idx` or when you reach `max_length`. SOS and EOS
guide the decoder, so do not include them in the IDS tokens you emit. Build this loop yourself;
do not call the checkpoint's built-in `greedy_search` or beam search.

Use the fixed fixtures in [`greedy_decode_fixtures.json`](greedy_decode_fixtures.json) to check
four important edge cases exactly: a normal EOS-terminated sequence, immediate EOS, repeated
emitted tokens, and an EOS-free sequence that stops at the maximum length.

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

While you decode, also keep `own`: the sum of the log-softmax probabilities of every token you
chose, including the stop token when the model chooses it. Step 2 needs it.

## Step 2: bidirectional decoding

The same decoder reads right-to-left when the prefix starts with `eos_idx`. In that direction it
predicts the **last** IDS token first and stops when it chooses `sos_idx`. Structure operators
such as `⿰` come first in an IDS, so right-to-left decoding predicts them last.

The two directions make different mistakes. The bidirectional decoder uses that:

1. **Decode both ways.** Run greedy decoding left-to-right (start `sos_idx`, stop `eos_idx`) and
   right-to-left (start `eos_idx`, stop `sos_idx`). Reverse the right-to-left tokens so both
   hypotheses are in natural IDS order. Record `own` for each.
2. **Re-score with the opposite direction.** For each hypothesis, compute `cross`: the
   teacher-forced log-probability of the same tokens read in the other direction, including
   that direction's stop token. For a left-to-right hypothesis `⿰木人`, `cross` is
   `log P(人 | EOS) + log P(木 | EOS 人) + log P(⿰ | EOS 人 木) + log P(SOS | EOS 人 木 ⿰)`.
   One teacher-forced forward pass gives all of these terms at once.
3. **Choose.** `score = (own + cross) / max(len(tokens), 1) ** alpha` with `alpha = 1.0`. The
   higher score wins. If the scores are exactly equal, keep the left-to-right result.

When both directions decode the **same** tokens, their scores are equal in exact arithmetic:
each is `log P_l2r(tokens) + log P_r2l(tokens)` divided by the same length. On a GPU, rounding
then decides the "winner", so your decoder and the reference notebook may label different
directions for these crops while returning identical tokens. Count such crops as agreement, not
as a win for either direction.

### Bidirectional fixtures

[`bidirectional_decode_fixtures.json`](bidirectional_decode_fixtures.json) has four tiny cases
over the vocabulary `<pad> <sos> <eos> ⿰ 木 人`. Each fixture maps a comma-separated prefix to the
next-token **probabilities**. Pass `log(p)` as the logits, which makes log-softmax the identity
and lets you check every number with a calculator.

| Fixture | What it checks |
| --- | --- |
| `r2l_wins` | Left-to-right repeats a token (`⿰木人人`); right-to-left gets `⿰木人` and wins on score. |
| `l2r_wins` | Right-to-left stops too early (`木`); its cross score is poor and left-to-right wins. |
| `agree_tie_keeps_l2r` | Both directions give `⿰木`; the scores tie exactly, so left-to-right is kept. |
| `r2l_max_length` | Right-to-left never stops and fills `max_length` with `人人人`, yet still wins after length normalization. |

Your decoder must reproduce every `tokens`, `best` and `own`/`cross`/`score` value (to 3
decimals) before you run it on real crops. Work through `r2l_wins` by hand first: its expected
right-to-left `own` is `log(0.8 × 0.8 × 0.9 × 0.9) = −0.657`.

The last fixture is deliberately uncomfortable. Dividing by length rewards long, confident
repeats. In the reference notebook, `說說說` and `來來` show the related effect at the lookup
step: one IDS maps to several characters. Keep both in mind when you inspect real outputs.

## IDS composition and output contract

Join the emitted vocabulary tokens into an IDS string, then look it up in the reverse
expanded-IDS mapping. One IDS can lead to several Unicode candidates, so preserve the whole
ordered list. If there is no direct Unicode match, keep the result and record an empty candidate
list; an unmatched sequence is still useful evidence.

Deduplicate before the lookup: many boxes on a page decode to the same token sequence, so look
up each unique sequence once and map the result back to every box.

Write JSONL or TSV records containing `page_id`, `box_id`, normalized and pixel boxes, detector
confidence, the winning direction, both directions' token IDs and scores, the chosen IDS string,
and Unicode candidates. Use the Week 4 prediction manifest as your input contract so every page
remains accounted for, including pages with no detector boxes.

For the full run, batch your crops: decode many crops per forward pass, as
`bidirectional_greedy` in the reference notebook does, rather than one crop at a time.

## Optional ablation: is the second direction worth it?

Dr. Cuong has not measured whether right-to-left decoding helps for IDS sequences, so this is an
open question. On the same set of crops, run three decoders: left-to-right only, right-to-left
only, and bidirectional. Report:

- how often the directions agree, and when they disagree, how often each one wins;
- agreement between each pair of decoders, and with beam search (`model.beam_search`) as a
  stronger reference;
- the share of outputs whose IDS has no Unicode candidate.

There are no character labels for these crops, so present the numbers as evidence about decoder
behaviour, not accuracy. A short slide on this is a good midterm addition.

## Tasks

- [ ] Implement and exact-match test left-to-right greedy decoding on all `greedy_decode_fixtures.json` cases.
- [ ] Extend it to bidirectional decoding and exact-match all `bidirectional_decode_fixtures.json` cases.
- [ ] Run Dr. Cuong's `evaluate.ipynb` and check that your decoder returns the same tokens on its crops.
- [ ] Implement token-sequence deduplication and IDS→Unicode candidate lookup.
- [ ] Run the full configured pipeline on the complete Week 4 detector handoff.
- [ ] Inspect representative correct-looking, ambiguous, and unmatched IDS outputs.
- [ ] Optional: the decoder ablation above.
- [ ] Prepare the midterm materials using [the presentation rules](../../docs/MIDTERM-PRESENTATION.md).

## Deliverable

A bidirectional decoder with fixture tests, full per-box recognition output, a combined detection
and recognition demo, and midterm slides. Keep predictions, weights, and private configuration out
of Git.
