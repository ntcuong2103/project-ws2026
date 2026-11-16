"""Small pure-Python helpers used to assess the Week 7 decoding exercise.

The bidirectional functions follow Dr. Cuong's reference notebook
(``nom-ids@project-2026/evaluate.ipynb``, section 2.1): decode greedily in both
directions, re-score each result with the opposite direction, and keep the
direction with the higher length-normalized score.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

NextLogits = Callable[[list[int]], Sequence[float]]
DIRECTIONS = ("l2r", "r2l")


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


def log_softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable log-softmax of one vocabulary row."""
    if not logits:
        raise ValueError("next_logits returned no vocabulary values")
    peak = max(logits)
    log_total = peak + math.log(sum(math.exp(value - peak) for value in logits))
    return [value - log_total for value in logits]


def _start_stop(direction: str, sos_idx: int, eos_idx: int) -> tuple[int, int]:
    if direction == "l2r":
        return sos_idx, eos_idx
    if direction == "r2l":
        return eos_idx, sos_idx
    raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")


@dataclass(frozen=True)
class DirectionalResult:
    """One direction's greedy hypothesis, always stored in natural (left-to-right) order."""

    direction: str
    tokens: tuple[int, ...]
    own: float
    cross: float
    score: float


def greedy_decode_direction(
    next_logits: NextLogits, *, direction: str, sos_idx: int, eos_idx: int, max_length: int
) -> tuple[tuple[int, ...], float]:
    """Greedy decoding in one direction.

    ``l2r`` starts from SOS and stops at EOS; ``r2l`` starts from EOS and stops at
    SOS, producing the IDS from its last token to its first. Returns the emitted
    tokens in natural order plus ``own``: the summed log-probability of every
    chosen token, including the stop token when it is chosen.
    """
    if max_length < 1:
        raise ValueError("max_length must be positive")
    start, stop = _start_stop(direction, sos_idx, eos_idx)
    prefix = [start]
    emitted: list[int] = []
    own = 0.0
    for _ in range(max_length):
        log_probs = log_softmax(next_logits(prefix))
        token = max(range(len(log_probs)), key=lambda index: log_probs[index])
        own += log_probs[token]
        if token == stop:
            break
        emitted.append(token)
        prefix.append(token)
    tokens = emitted[::-1] if direction == "r2l" else emitted
    return tuple(tokens), own


def sequence_log_prob(
    next_logits: NextLogits, tokens: Sequence[int], *, direction: str, sos_idx: int, eos_idx: int
) -> float:
    """Teacher-forced log P(tokens) read in ``direction``, including the stop token.

    ``tokens`` are in natural order. For ``r2l`` they are read backwards. Because
    the decoder is causal, calling ``next_logits`` on each growing prefix gives
    the same numbers as one teacher-forced forward pass over the whole target.
    """
    start, stop = _start_stop(direction, sos_idx, eos_idx)
    ordered = list(tokens)[::-1] if direction == "r2l" else list(tokens)
    prefix = [start]
    total = 0.0
    for target in ordered + [stop]:
        total += log_softmax(next_logits(prefix))[target]
        prefix.append(target)
    return total


def bidirectional_greedy_decode(
    next_logits: NextLogits, *, sos_idx: int, eos_idx: int, max_length: int, alpha: float = 1.0
) -> dict[str, object]:
    """Bidirectional greedy decoding with cross-direction re-scoring.

    ``score = (own + cross) / max(len(tokens), 1) ** alpha``. The higher score
    wins; a tie keeps ``l2r``, matching the reference notebook.
    """
    opposite = {"l2r": "r2l", "r2l": "l2r"}
    results: dict[str, DirectionalResult] = {}
    for direction in DIRECTIONS:
        tokens, own = greedy_decode_direction(
            next_logits, direction=direction, sos_idx=sos_idx, eos_idx=eos_idx, max_length=max_length
        )
        cross = sequence_log_prob(
            next_logits, tokens, direction=opposite[direction], sos_idx=sos_idx, eos_idx=eos_idx
        )
        score = (own + cross) / max(len(tokens), 1) ** alpha
        results[direction] = DirectionalResult(direction, tokens, own, cross, score)
    best = max(DIRECTIONS, key=lambda direction: results[direction].score)
    return {"l2r": results["l2r"], "r2l": results["r2l"], "best": best, "tokens": results[best].tokens}
