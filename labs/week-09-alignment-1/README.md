# Week 9 — Alignment I: Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 8](../week-08-midterm/) · Next: [Week 10](../week-10-alignment-2/)

**Session:** each group reviews Week 7 and the midterm feedback (5 min/group), then this lab.

## Goals

- Formalize correspondence recovery between detected boxes and ground-truth lines.
- Implement box-to-line assignment from the paper's description alone.

## Starter kit (given)

None — this is core to the task.

## You implement

The center-in-rectangle box-to-line test and the box-to-line grouping/sorting logic.

## Reading

Nguyen et al., "Automated Character-Level Annotation...," ICDAR (2026) — the paper's own prior work. Source paper §3.2.

## Small experiment

Build 2 synthetic pages by hand (a handful of boxes, 2–3 lines) where you know which box belongs to which line. Your function must match before you run it on real pages.

## Tutorial steps

1. Define the line regions and the center-in-rectangle test.
2. Group boxes by line and sort within each line (reading order for the script).
3. Validate on the 2 synthetic pages.
4. Run on the team's real detected boxes and ground-truth lines.
5. Write up 2–3 edge cases and your decisions (e.g. a box on a line boundary).

## Tasks

- [ ] Implement box-to-line assignment from the paper's §3.2 description
- [ ] Validate on 2 hand-built synthetic pages
- [ ] Run it on the team's real detected boxes and ground-truth lines
- [ ] Write up 2–3 edge cases you had to decide on (e.g., a box on a line boundary)

## Deliverable

Box-to-line assignment module, validated on synthetic pages, then run on real data.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
