"""Small pure-Python helpers used to assess the Week 7 decoding exercise."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Sequence


def greedy_decode(next_logits: Callable[[list[int]], Sequence[float]], *, sos_idx: int, eos_idx: int, max_length: int) -> list[int]:
    """Decode one sequence, excluding SOS/EOS from the returned IDS token IDs."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    prefix = [sos_idx]
    emitted: list[int] = []
    for _ in range(max_length):
        logits = next_logits(prefix)
        if not logits:
            raise ValueError("next_logits returned no vocabulary values")
        token = max(range(len(logits)), key=lambda index: logits[index])
        if token == eos_idx:
            break
        emitted.append(token)
        prefix.append(token)
    return emitted


def deduplicate_sequences(sequences: Iterable[Sequence[int]]) -> tuple[list[tuple[int, ...]], list[int]]:
    """Return ordered unique sequences and an index mapping for every input."""
    unique: list[tuple[int, ...]] = []
    lookup: dict[tuple[int, ...], int] = {}
    mapping = []
    for sequence in sequences:
        key = tuple(sequence)
        mapping.append(lookup.setdefault(key, len(unique)))
        if mapping[-1] == len(unique):
            unique.append(key)
    return unique, mapping


def ids_candidates(ids: str, reverse_mapping: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Return all composed Unicode candidates; an unknown IDS has no candidates."""
    return reverse_mapping.get(ids, ())
