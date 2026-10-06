# Week 5 — Recognition I: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 4](../week-04-detection-2/) · Next: [Week 6](../week-06-recognition-2/)

**Session:** each group reviews Week 4 (5 min/group), then this lab.

## Goals

- Understand Ideographic Description Sequences (IDS) and compositional character representation.
- Implement the text→IDS mapping and measure vocabulary coverage on the team's subset.

## Starter kit (given)

The character→IDS dictionary (CHISE/cjkvi-ids) and vocabulary files only — no decomposition code.

## You implement

A function mapping ground-truth line text to IDS strings, character by character.

## Reading

Zhang, Du & Dai, "Radical analysis network..." (2020); Morioka, "CHISE" (2008).

## Small experiment

Pick a 5-character line and write down the expected IDS strings by looking them up in the dictionary. Your function must reproduce them exactly before you run it on the full subset.

## Tutorial steps

1. Inspect the IDS dictionary format: operators (e.g. ⿰, ⿱), components, characters with multiple decompositions.
2. Implement the text→IDS mapping.
3. Validate on the 5-character example.
4. Run over the full assigned subset; report vocabulary coverage %.
5. Prepare the cropped character-image set for recognizer inference.

## Tasks

- [ ] Implement the text→IDS mapping function
- [ ] Validate it on the 5-character hand-picked example
- [ ] Run it over the team's full assigned subset; report vocabulary coverage %
- [ ] Prepare the cropped character-image set needed for recognizer inference

## Deliverable

IDS decomposition utility (validated on a toy example) + recognizer training set.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
