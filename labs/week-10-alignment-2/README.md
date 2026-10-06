# Week 10 — Alignment II (the core algorithm): Lab Tutorial

[← Back to course README](../../README.md) · Previous: [Week 9](../week-09-alignment-1/) · Next: [Week 11](../week-11-alignment-3/)

**Session:** each group reviews Week 9 (5 min/group), then this lab.

## Goals

- Implement the two-level IDS-level dynamic-programming alignment — the paper's Algorithm 1.
- Verify it against a table you computed by hand.

## Starter kit (given)

- A function stub with the paper's signature/docstring (`raise NotImplementedError`).
- A plain Levenshtein `edit_distance()` utility (implement it yourself first if you don't already have it from Week 9).

## You implement

The full two-level DP alignment: outer character-to-character alignment scored by inner IDS edit distance, length-proportional gap costs, and backtracking to recover the correspondence map (Algorithm 1, Eq. 1).

## Small experiment

A hand-built toy example: 5–6 predicted boxes' IDS strings vs. 5–6 ground-truth characters' IDS strings. Compute the expected DP table and correspondence by hand first, then check your code reproduces it exactly.

## Tutorial steps

1. Work the toy example by hand: fill the DP table and trace back the alignment.
2. Implement `edit_distance()` if needed (the inner level).
3. Implement the outer DP table with gap costs proportional to length.
4. Implement backtracking to recover the correspondence map.
5. Compare with your hand-computed table; fix any discrepancy.
6. Write the two-level cost model in your own words (outer vs. inner).

## Tasks

- [ ] Implement `edit_distance()` (if needed) and the outer DP table + backtracking per Algorithm 1
- [ ] Hand-compute the expected DP table and alignment for the toy example
- [ ] Verify your code matches the hand-computed result exactly; fix discrepancies
- [ ] Document the two-level cost model in your own words (outer vs. inner)

## Deliverable

DP alignment implementation, verified against a hand-computed toy example.

## Materials

_Add notebooks, scripts and fixtures for this lab in this folder._
