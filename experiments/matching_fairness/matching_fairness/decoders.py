"""Label-free deterministic decoders for retrieval matching experiments."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


@dataclass(frozen=True)
class Assignment:
    """One gallery decision per query; ``-1`` denotes an unmatched query."""

    gallery_indices: np.ndarray
    unmatched_mask: np.ndarray
    strict_one_to_one: bool
    metadata: Mapping[str, object]


def _validated_matrix(similarity: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(similarity, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("similarity must be a non-empty 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("similarity contains NaN or Inf")
    return matrix


def _assignment(
    gallery_indices: np.ndarray,
    *,
    strict_one_to_one: bool,
    metadata: Mapping[str, object] | None = None,
) -> Assignment:
    indices = np.asarray(gallery_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("gallery assignment must be one-dimensional")
    indices.setflags(write=False)
    unmatched = indices < 0
    unmatched.setflags(write=False)
    return Assignment(
        gallery_indices=indices,
        unmatched_mask=unmatched,
        strict_one_to_one=strict_one_to_one,
        metadata=dict(metadata or {}),
    )

def decode_independent(similarity: np.ndarray) -> Assignment:
    matrix = _validated_matrix(similarity)
    return _assignment(np.argmax(matrix, axis=1), strict_one_to_one=False)


def decode_greedy(similarity: np.ndarray) -> Assignment:
    matrix = _validated_matrix(similarity)
    rows, columns = matrix.shape
    ranking = np.argsort(-matrix, axis=1, kind="stable")
    top1 = ranking[:, 0]
    top_scores = matrix[np.arange(rows), top1]
    row_order = np.lexsort((np.arange(rows), -top_scores))
    indices = -np.ones(rows, dtype=np.int64)
    used = np.zeros(columns, dtype=bool)
    for row in row_order:
        gallery = int(top1[row])
        if not used[gallery]:
            indices[int(row)] = gallery
            used[gallery] = True
    for row in np.flatnonzero(indices < 0):
        available = ranking[row][~used[ranking[row]]]
        if available.size:
            gallery = int(available[0])
            indices[int(row)] = gallery
            used[gallery] = True
    matched = indices >= 0
    return _assignment(
        indices,
        strict_one_to_one=True,
        metadata={
            "matched_count": int(matched.sum()),
            "unmatched_count": int((~matched).sum()),
        },
    )


def decode_hungarian(similarity: np.ndarray, seed: int) -> Assignment:
    matrix = _validated_matrix(similarity)
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Hungarian seed must be an integer")
    rows, columns = matrix.shape
    generator = np.random.default_rng(int(seed))
    row_permutation = generator.permutation(rows)
    column_permutation = generator.permutation(columns)
    selected_rows, selected_columns = linear_sum_assignment(
        matrix[row_permutation][:, column_permutation], maximize=True
    )
    indices = -np.ones(rows, dtype=np.int64)
    indices[row_permutation[selected_rows]] = column_permutation[selected_columns]
    matched = indices >= 0
    return _assignment(
        indices,
        strict_one_to_one=True,
        metadata={
            "seed": int(seed),
            "matched_count": int(matched.sum()),
            "unmatched_count": int((~matched).sum()),
            "assigned_sum_similarity": float(
                matrix[np.flatnonzero(matched), indices[matched]].sum()
            ),
        },
    )


def decode_stable(similarity: np.ndarray) -> Assignment:
    matrix = _validated_matrix(similarity)
    rows, columns = matrix.shape
    preferences = np.argsort(-matrix, axis=1, kind="stable")
    next_choice = np.zeros(rows, dtype=np.int64)
    gallery_partner = -np.ones(columns, dtype=np.int64)
    indices = -np.ones(rows, dtype=np.int64)
    free_queries: deque[int] = deque(range(rows))
    while free_queries:
        query = free_queries.popleft()
        if next_choice[query] >= columns:
            continue
        gallery = int(preferences[query, next_choice[query]])
        next_choice[query] += 1
        current = int(gallery_partner[gallery])
        if current < 0:
            gallery_partner[gallery] = query
            indices[query] = gallery
            continue
        challenger_wins = matrix[query, gallery] > matrix[current, gallery] or (
            matrix[query, gallery] == matrix[current, gallery] and query < current
        )
        if challenger_wins:
            indices[current] = -1
            if next_choice[current] < columns:
                free_queries.append(current)
            gallery_partner[gallery] = query
            indices[query] = gallery
        elif next_choice[query] < columns:
            free_queries.append(query)
    matched = indices >= 0
    return _assignment(
        indices,
        strict_one_to_one=True,
        metadata={
            "matched_count": int(matched.sum()),
            "unmatched_count": int((~matched).sum()),
            "proposal_count": int(next_choice.sum()),
        },
    )


def decode_sinkhorn(
    similarity: np.ndarray,
    temperature: float,
    max_iterations: int,
    tolerance: float,
) -> Assignment:
    matrix = _validated_matrix(similarity)
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, (int, np.integer))
        or not np.isfinite(temperature)
        or not np.isfinite(tolerance)
        or temperature <= 0
        or max_iterations < 1
        or tolerance <= 0
    ):
        raise ValueError("invalid Sinkhorn parameters")
    rows, columns = matrix.shape
    log_kernel = matrix / float(temperature)
    log_u = np.zeros(rows, dtype=np.float64)
    log_v = np.zeros(columns, dtype=np.float64)
    marginal_error = float("inf")
    converged = False
    for iteration in range(1, int(max_iterations) + 1):
        log_u = -np.log(float(rows)) - logsumexp(
            log_kernel + log_v[None, :], axis=1
        )
        log_v = -np.log(float(columns)) - logsumexp(
            log_kernel + log_u[:, None], axis=0
        )
        if iteration == 1 or iteration % 10 == 0 or iteration == max_iterations:
            plan = np.exp(log_u[:, None] + log_kernel + log_v[None, :])
            row_error = np.max(np.abs(plan.sum(axis=1) - 1.0 / rows))
            column_error = np.max(np.abs(plan.sum(axis=0) - 1.0 / columns))
            marginal_error = float(max(row_error, column_error))
            if marginal_error <= tolerance:
                converged = True
                break
    plan = np.ascontiguousarray(
        np.exp(log_u[:, None] + log_kernel + log_v[None, :]),
        dtype=np.float64,
    )
    if not np.isfinite(plan).all():
        raise RuntimeError("Sinkhorn produced a non-finite transport plan")
    plan_digest = hashlib.sha256()
    plan_digest.update(np.asarray(plan.shape, dtype=np.int64).tobytes())
    plan_digest.update(plan.tobytes(order="C"))
    return _assignment(
        np.argmax(plan, axis=1),
        strict_one_to_one=False,
        metadata={
            "temperature": float(temperature),
            "max_iterations": int(max_iterations),
            "iterations": int(iteration),
            "tolerance": float(tolerance),
            "marginal_error": marginal_error,
            "converged": converged,
            "plan_min": float(plan.min()),
            "plan_max": float(plan.max()),
            "plan_sum": float(plan.sum()),
            "plan_sha256": plan_digest.hexdigest(),
        },
    )
