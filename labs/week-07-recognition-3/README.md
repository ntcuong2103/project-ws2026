# Week 7 — Recognition III: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 6](../week-06-recognition-2/) · Next: [Week 8](../week-08-midterm/)

**Session:** each group reviews Week 6 (5 min/group), then this lab.

## Goals

- Speed up decoding by deduplicating identical token sequences.
- Map decoded IDS strings back to composed Unicode characters where possible.
- Assemble the combined detection + recognition demo and prepare midterm slides.

## Starter kit (given)

None new.

## You implement

Deduplication of identical token sequences before decoding, and IDS→Unicode composition.

## Small experiment

Test both functions on a handful of hand-picked token sequences — including one repeated sequence and one IDS string with no direct Unicode match — with known expected outputs.

## Tutorial steps

1. Implement dedup: decode each unique sequence once, map results back to all boxes.
2. Implement IDS→Unicode composition; decide what to return when there is no direct match.
3. Validate both on the hand-picked examples.
4. Run the full decode step on the team's subset: predicted text + IDS per box.
5. Build the combined demo and draft the midterm slides.

## Tasks

- [ ] Implement sequence deduplication and IDS→Unicode composition
- [ ] Validate both on the hand-picked examples
- [ ] Run the full decode step on the team's subset to produce predicted text + IDS per box
- [ ] Assemble a combined detection+recognition demo; draft midterm slides

## Deliverable

Decoding module (validated on toy examples) + combined demo + midterm slide draft.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
