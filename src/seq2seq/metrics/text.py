from __future__ import annotations

from collections import Counter
from math import exp, log
from typing import Sequence

TokenSeq = Sequence[str]


def _ngrams(tokens: TokenSeq, n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(
    hypothesis: TokenSeq,
    references: Sequence[TokenSeq],
    n: int,
) -> tuple[int, int]:
    hyp_ngrams = _ngrams(hypothesis, n)
    if not hyp_ngrams:
        return 0, 0
    max_ref: Counter = Counter()
    for reference in references:
        ref_ngrams = _ngrams(reference, n)
        for ngram, count in ref_ngrams.items():
            if count > max_ref[ngram]:
                max_ref[ngram] = count
    clipped = sum(min(count, max_ref[ngram]) for ngram, count in hyp_ngrams.items())
    total = sum(hyp_ngrams.values())
    return clipped, total


def _closest_reference_length(references: Sequence[TokenSeq], hyp_len: int) -> int:
    if not references:
        return 0
    return min(
        (len(reference) for reference in references),
        key=lambda ref_len: (abs(ref_len - hyp_len), ref_len),
    )


def corpus_bleu(
    hypotheses: Sequence[TokenSeq],
    references: Sequence[Sequence[TokenSeq]],
    *,
    max_n: int = 4,
    weights: Sequence[float] | None = None,
    smoothing_epsilon: float = 1e-9,
) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must be the same length")
    if weights is None:
        weights = [1.0 / max_n] * max_n
    if len(weights) != max_n:
        raise ValueError("weights length must equal max_n")

    clipped_totals = [0] * max_n
    count_totals = [0] * max_n
    hyp_length_total = 0
    ref_length_total = 0

    for hypothesis, refs in zip(hypotheses, references):
        hypothesis = list(hypothesis)
        refs = [list(ref) for ref in refs]
        hyp_length_total += len(hypothesis)
        ref_length_total += _closest_reference_length(refs, len(hypothesis))
        for n in range(1, max_n + 1):
            clipped, total = _modified_precision(hypothesis, refs, n)
            clipped_totals[n - 1] += clipped
            count_totals[n - 1] += total

    log_score = 0.0
    for n in range(max_n):
        numerator = clipped_totals[n]
        denominator = count_totals[n]
        if denominator == 0:
            return 0.0
        precision = (numerator + smoothing_epsilon) / (denominator + smoothing_epsilon)
        log_score += weights[n] * log(precision)

    if hyp_length_total == 0:
        return 0.0
    if hyp_length_total > ref_length_total:
        brevity_penalty = 1.0
    else:
        brevity_penalty = exp(1.0 - ref_length_total / hyp_length_total)
    return float(brevity_penalty * exp(log_score))


def bleu_score(
    hypothesis: TokenSeq,
    references: Sequence[TokenSeq],
    *,
    max_n: int = 4,
    weights: Sequence[float] | None = None,
) -> float:
    return corpus_bleu([hypothesis], [references], max_n=max_n, weights=weights)


def _count_chunks(matches: Sequence[tuple[int, int]]) -> int:
    if not matches:
        return 0
    ordered = sorted(matches, key=lambda pair: pair[0])
    chunks = 1
    for index in range(1, len(ordered)):
        prev_hyp, prev_ref = ordered[index - 1]
        curr_hyp, curr_ref = ordered[index]
        if curr_hyp - prev_hyp != 1 or curr_ref - prev_ref != 1:
            chunks += 1
    return chunks


def _align_exact(hypothesis: TokenSeq, reference: TokenSeq) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    used_ref = [False] * len(reference)
    for h_index, token in enumerate(hypothesis):
        best: int | None = None
        for r_index, ref_token in enumerate(reference):
            if used_ref[r_index] or ref_token != token:
                continue
            if best is None:
                best = r_index
                break
        if best is not None:
            used_ref[best] = True
            matches.append((h_index, best))
    return matches


def _meteor_single(
    hypothesis: TokenSeq,
    reference: TokenSeq,
    *,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    if not hypothesis or not reference:
        return 0.0
    matches = _align_exact(list(hypothesis), list(reference))
    if not matches:
        return 0.0
    match_count = len(matches)
    precision = match_count / len(hypothesis)
    recall = match_count / len(reference)
    denom = alpha * precision + (1.0 - alpha) * recall
    if denom == 0:
        return 0.0
    fmean = (precision * recall) / denom
    chunks = _count_chunks(matches)
    penalty = gamma * (chunks / match_count) ** beta
    return float(fmean * (1.0 - penalty))


def meteor_score(
    hypothesis: TokenSeq,
    references: Sequence[TokenSeq],
    *,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    if not references:
        return 0.0
    return max(
        _meteor_single(hypothesis, reference, alpha=alpha, beta=beta, gamma=gamma)
        for reference in references
    )


__all__ = ["bleu_score", "corpus_bleu", "meteor_score"]
