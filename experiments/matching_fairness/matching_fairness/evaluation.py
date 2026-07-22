"""Canonical-ID evaluation shared by every label-free decoder."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import re

import numpy as np

from .artifacts import ScoreArtifact, independent_ranks
from .decoders import (
    Assignment,
    decode_greedy,
    decode_hungarian,
    decode_independent,
    decode_sinkhorn,
    decode_stable,
)


DECODER_NAMES = (
    "independent",
    "greedy",
    "hungarian",
    "stable_matching",
    "sinkhorn",
)


@dataclass(frozen=True)
class DecoderConfig:
    """Fixed decoder choice and pre-registered non-label parameters."""

    name: str
    seed: int = 42
    temperature: float = 0.05
    max_iterations: int = 500
    tolerance: float = 1e-8


@dataclass(frozen=True)
class EvaluationResult:
    """Summary and per-query ledger for one artifact/decoder cell."""

    decoder: str
    assignment: Assignment
    metrics: Mapping[str, object]
    per_query: tuple[Mapping[str, object], ...]


def evaluate_artifact(
    artifact: ScoreArtifact,
    decoder_config: DecoderConfig,
) -> EvaluationResult:
    """Decode without labels, then score predictions by canonical identity."""

    artifact.validate()
    if not isinstance(decoder_config, DecoderConfig):
        raise ValueError("decoder_config must be a DecoderConfig")
    assignment = _decode(artifact.similarity, decoder_config)
    rows, columns = artifact.similarity.shape
    if assignment.gallery_indices.shape != (rows,):
        raise ValueError("decoder assignment shape does not match artifact queries")
    if np.any(assignment.gallery_indices >= columns):
        raise ValueError("decoder returned an out-of-range gallery index")
    matched_indices = assignment.gallery_indices[~assignment.unmatched_mask]
    if assignment.strict_one_to_one and len(set(matched_indices.tolist())) != len(
        matched_indices
    ):
        raise ValueError("strict one-to-one decoder returned duplicate entries")

    gallery_set = set(artifact.gallery_canonical_ids)
    answerable = np.fromiter(
        (target in gallery_set for target in artifact.target_canonical_ids),
        dtype=bool,
        count=rows,
    )
    predicted_canonical: list[str | None] = []
    predicted_entry: list[str | None] = []
    assigned_scores: list[float | None] = []
    correct = np.zeros(rows, dtype=bool)
    for row, gallery_index in enumerate(assignment.gallery_indices):
        if gallery_index < 0:
            predicted_canonical.append(None)
            predicted_entry.append(None)
            assigned_scores.append(None)
            continue
        gallery = int(gallery_index)
        canonical_id = artifact.gallery_canonical_ids[gallery]
        predicted_canonical.append(canonical_id)
        predicted_entry.append(artifact.gallery_entry_ids[gallery])
        assigned_scores.append(float(artifact.similarity[row, gallery]))
        correct[row] = canonical_id == artifact.target_canonical_ids[row]

    top5_mask: np.ndarray | None = None
    if decoder_config.name == "independent":
        top5_mask = answerable & (independent_ranks(artifact) <= 5)

    independent = decode_independent(artifact.similarity)
    independent_correct = np.fromiter(
        (
            artifact.gallery_canonical_ids[int(gallery)]
            == artifact.target_canonical_ids[row]
            for row, gallery in enumerate(independent.gallery_indices)
        ),
        dtype=bool,
        count=rows,
    )
    transitions = {
        "correct_to_correct": int(np.count_nonzero(independent_correct & correct)),
        "correct_to_wrong": int(np.count_nonzero(independent_correct & ~correct)),
        "wrong_to_correct": int(np.count_nonzero(~independent_correct & correct)),
        "wrong_to_wrong": int(np.count_nonzero(~independent_correct & ~correct)),
    }

    correct_count = int(correct.sum())
    answerable_count = int(answerable.sum())
    answerable_correct = int(np.count_nonzero(correct & answerable))
    matched_count = int((~assignment.unmatched_mask).sum())
    metrics: dict[str, object] = {
        "decoder": decoder_config.name,
        "correct": correct_count,
        "total": rows,
        "top1": _percent(correct_count, rows),
        "answerable_correct": answerable_correct,
        "answerable_total": answerable_count,
        "answerable_top1": _percent(answerable_correct, answerable_count),
        "unanswerable_count": rows - answerable_count,
        "assigned_count": matched_count,
        "unmatched_count": rows - matched_count,
        "unique_gallery_entry_predictions": len(
            set(int(value) for value in matched_indices)
        ),
        "unique_canonical_predictions": len(
            {value for value in predicted_canonical if value is not None}
        ),
        "strict_one_to_one": assignment.strict_one_to_one,
        "top5_count": int(top5_mask.sum()) if top5_mask is not None else None,
        "top5": _percent(int(top5_mask.sum()), rows)
        if top5_mask is not None
        else None,
        "assignment_changes_from_independent": int(
            np.count_nonzero(
                assignment.gallery_indices != independent.gallery_indices
            )
        ),
        "delta_correct_vs_independent": correct_count
        - int(independent_correct.sum()),
        "assignment_metadata": dict(assignment.metadata),
        **transitions,
    }
    duplicate_metrics = _duplicate_query_metrics(
        artifact,
        assignment,
        correct,
        answerable,
    )
    metrics.update(duplicate_metrics)

    per_query = tuple(
        {
            "query_index": row,
            "query_id": artifact.query_ids[row],
            "target_canonical_id": artifact.target_canonical_ids[row],
            "answerable": bool(answerable[row]),
            "gallery_index": int(assignment.gallery_indices[row]),
            "predicted_gallery_entry_id": predicted_entry[row],
            "predicted_canonical_id": predicted_canonical[row],
            "assigned_score": assigned_scores[row],
            "unmatched": bool(assignment.unmatched_mask[row]),
            "correct_top1": bool(correct[row]),
            "correct_top5": bool(top5_mask[row])
            if top5_mask is not None
            else None,
        }
        for row in range(rows)
    )
    return EvaluationResult(
        decoder=decoder_config.name,
        assignment=assignment,
        metrics=metrics,
        per_query=per_query,
    )


def _decode(similarity: np.ndarray, config: DecoderConfig) -> Assignment:
    if config.name not in DECODER_NAMES:
        raise ValueError(f"unsupported formal decoder: {config.name}")
    if config.name == "independent":
        return decode_independent(similarity)
    if config.name == "greedy":
        return decode_greedy(similarity)
    if config.name == "hungarian":
        return decode_hungarian(similarity, seed=config.seed)
    if config.name == "stable_matching":
        return decode_stable(similarity)
    return decode_sinkhorn(
        similarity,
        temperature=config.temperature,
        max_iterations=config.max_iterations,
        tolerance=config.tolerance,
    )


def _duplicate_query_metrics(
    artifact: ScoreArtifact,
    assignment: Assignment,
    correct: np.ndarray,
    answerable: np.ndarray,
) -> dict[str, object]:
    mode = artifact.metadata.get("query_mode")
    if mode is None:
        return {}
    match = re.fullmatch(r"dupq(0|10|20)", str(mode))
    if match is None:
        raise ValueError("duplicate-query artifact has invalid query_mode")
    expected_repeats = int(match.group(1))
    rows_by_target: dict[str, list[int]] = defaultdict(list)
    for row, target in enumerate(artifact.target_canonical_ids):
        rows_by_target[target].append(row)
    repeated = {
        target: tuple(rows)
        for target, rows in rows_by_target.items()
        if len(rows) > 1
    }
    if len(repeated) != expected_repeats or any(
        len(rows) != 2 for rows in repeated.values()
    ):
        raise ValueError("duplicate-query canonical multiplicities do not match query_mode")

    appended_b = np.fromiter(
        (query_id.endswith("__eeg_b") for query_id in artifact.query_ids),
        dtype=bool,
        count=len(artifact.query_ids),
    )
    if int(appended_b.sum()) != expected_repeats:
        raise ValueError("duplicate-query artifact does not identify every EEG-B row")
    repeated_targets = set(repeated)
    if {
        artifact.target_canonical_ids[row]
        for row in np.flatnonzero(appended_b)
    } != repeated_targets:
        raise ValueError("EEG-B rows do not match repeated canonical targets")
    base_a = ~appended_b

    at_least_one = sum(
        any(bool(correct[row]) for row in rows) for rows in repeated.values()
    )
    both = sum(
        all(bool(correct[row]) for row in rows) for rows in repeated.values()
    )
    query_counts = Counter(artifact.target_canonical_ids)
    gallery_counts = Counter(artifact.gallery_canonical_ids)
    if assignment.strict_one_to_one:
        ceiling = sum(
            min(query_count, gallery_counts.get(target, 0))
            for target, query_count in query_counts.items()
        )
    else:
        ceiling = int(answerable.sum())
    correct_count = int(correct.sum())
    repeated_row_mask = np.fromiter(
        (target in repeated_targets for target in artifact.target_canonical_ids),
        dtype=bool,
        count=len(artifact.target_canonical_ids),
    )
    base_total = int(base_a.sum())
    b_total = int(appended_b.sum())
    return {
        "base_a_correct": int(np.count_nonzero(correct & base_a)),
        "base_a_total": base_total,
        "base_a_top1": _percent(int(np.count_nonzero(correct & base_a)), base_total),
        "appended_b_correct": int(np.count_nonzero(correct & appended_b)),
        "appended_b_total": b_total,
        "appended_b_top1": _percent(
            int(np.count_nonzero(correct & appended_b)), b_total
        ),
        "repeated_canonical_total": len(repeated),
        "at_least_one_correct_count": at_least_one,
        "at_least_one_coverage": _percent(at_least_one, len(repeated)),
        "both_correct_count": both,
        "both_correct": _percent(both, len(repeated)),
        "theoretical_ceiling_count": int(ceiling),
        "theoretical_ceiling": _percent(int(ceiling), len(correct)),
        "distance_from_ceiling": int(ceiling) - correct_count,
        "unmatched_repeated_queries": int(
            np.count_nonzero(assignment.unmatched_mask & repeated_row_mask)
        ),
    }


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return 100.0 * float(numerator) / float(denominator)
