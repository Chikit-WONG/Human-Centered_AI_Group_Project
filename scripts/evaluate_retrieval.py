#!/usr/bin/env python3
"""Evaluate a final Brain-RW checkpoint on THINGS-EEG image retrieval."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import datasets
import numpy as np
import peft
import scipy
import torch
import torch.nn.functional as F
import transformers
from peft import PeftModel
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MATCHING_FAIRNESS_ROOT = PROJECT_ROOT / "experiments/matching_fairness"
if str(MATCHING_FAIRNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(MATCHING_FAIRNESS_ROOT))

from main.data import (  # noqa: E402
    load_image_dataset,
    load_things_brain_dataset,
    merge_datasets_by_image_id,
)
from main.models_clip import BrainCLIPModel  # noqa: E402
from matching_fairness.artifacts import (  # noqa: E402
    ScoreArtifact,
    independent_ranks,
    write_score_artifact,
)

HUNGARIAN_PRIMARY_ORDER_SEED = 3800
HUNGARIAN_AUDIT_ORDER_SEEDS = tuple(range(3801, 3809))


def build_brainrw_score_artifact(
    *,
    similarity: np.ndarray,
    query_embeddings: np.ndarray,
    query_ids: Sequence[str],
    gallery_ids: Sequence[str],
    trial_half: str,
    brain_model_path: Path,
    top1_count: int,
    top5_count: int,
) -> ScoreArtifact:
    """Build a BrainRW artifact and verify canonical-ID metric parity."""
    if trial_half not in {"standard", "a", "b"}:
        raise ValueError("trial_half must be standard, a, or b")
    query_ids = tuple(query_ids)
    gallery_ids = tuple(gallery_ids)
    artifact = ScoreArtifact(
        similarity=np.ascontiguousarray(similarity, dtype=np.float32),
        query_ids=query_ids,
        gallery_entry_ids=gallery_ids,
        gallery_canonical_ids=gallery_ids,
        target_canonical_ids=query_ids,
        metadata={
            "model_slug": "our_project",
            "trial_half": trial_half,
            "checkpoint_role": "fixed_formal",
            "checkpoint": str(Path(brain_model_path)),
            "similarity": "cosine",
            "query_embeddings_sha256": sha256_array(query_embeddings),
        },
    )
    ranks = independent_ranks(artifact)
    metrics = {
        "top1_count": int(np.count_nonzero(ranks <= 1)),
        "top5_count": int(np.count_nonzero(ranks <= 5)),
        "sample_count": len(ranks),
    }
    expected = {
        "top1_count": top1_count,
        "top5_count": top5_count,
        "sample_count": len(query_ids),
    }
    if metrics != expected:
        raise ValueError(
            f"BrainRW ScoreArtifact metric parity failed: {metrics} != {expected}"
        )
    metadata = dict(artifact.metadata)
    metadata["native_metrics"] = metrics
    artifact = ScoreArtifact(
        similarity=artifact.similarity,
        query_ids=artifact.query_ids,
        gallery_entry_ids=artifact.gallery_entry_ids,
        gallery_canonical_ids=artifact.gallery_canonical_ids,
        target_canonical_ids=artifact.target_canonical_ids,
        metadata=metadata,
    )
    artifact.validate()
    return artifact


def load_trial_indices_by_image(
    path: Path, half: str
) -> dict[str, tuple[int, ...]]:
    """Load one canonical 40-trial half from a Task 4 manifest."""
    if half not in {"a", "b"}:
        raise ValueError("trial half must be a or b")
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid trial split manifest: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("trial split manifest must use schema_version 1")
    images = payload.get("images")
    image_ids = payload.get("image_ids")
    if (
        not isinstance(images, Mapping)
        or not isinstance(image_ids, list)
        or not image_ids
        or any(not isinstance(value, str) or not value for value in image_ids)
        or len(set(image_ids)) != len(image_ids)
        or set(images) != set(image_ids)
    ):
        raise ValueError("trial split manifest has invalid canonical image IDs")

    result: dict[str, tuple[int, ...]] = {}
    for image_id in image_ids:
        sessions = images[image_id]
        if not isinstance(sessions, Mapping) or len(sessions) != 4:
            raise ValueError("each manifest image must contain exactly 4 sessions")
        selected: list[int] = []
        all_indices: list[int] = []
        for split in sessions.values():
            if not isinstance(split, Mapping):
                raise ValueError("manifest session split must be a mapping")
            halves: dict[str, tuple[int, ...]] = {}
            for name in ("a", "b"):
                value = split.get(name)
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                    raise ValueError("manifest trial indices must be sequences")
                indices = tuple(value)
                if (
                    len(indices) != 10
                    or any(
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or not 0 <= index < 80
                        for index in indices
                    )
                    or len(set(indices)) != len(indices)
                ):
                    raise ValueError(
                        "each manifest session half must contain 10 unique "
                        "in-range trial indices"
                    )
                halves[name] = indices
            if set(halves["a"]).intersection(halves["b"]):
                raise ValueError("manifest session halves must not overlap")
            selected.extend(halves[half])
            all_indices.extend(halves["a"])
            all_indices.extend(halves["b"])
        if len(selected) != 40 or len(set(selected)) != 40:
            raise ValueError("manifest half must select exactly 40 unique trials")
        if len(all_indices) != 80 or len(set(all_indices)) != 80:
            raise ValueError("manifest sessions must account for 80 distinct trials")
        result[image_id] = tuple(selected)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final-checkpoint 200-way EEG-to-image retrieval."
    )
    parser.add_argument("--brain-model-path", required=True)
    parser.add_argument("--vision-adapter-path", required=True)
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--brain-directory", required=True)
    parser.add_argument("--image-directory", required=True)
    parser.add_argument("--dataset-name", default="things")
    parser.add_argument("--subject-id", type=int, default=8)
    parser.add_argument("--selected-channels", required=True)
    parser.add_argument("--time-slice", default="0,250")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bf16"), default="bf16"
    )
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-num-samples", type=int, default=200)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument(
        "--enable-hungarian",
        action="store_true",
        help=(
            "Also evaluate a transductive global one-to-one assignment. "
            "This is separate from standard per-query Top-k retrieval."
        ),
    )
    parser.add_argument(
        "--hungarian-output",
        help="Per-query CSV for the global one-to-one assignment.",
    )
    parser.add_argument(
        "--similarity-output",
        help="Auditable NPZ containing the full similarity matrix and ID order.",
    )
    parser.add_argument("--expected-top1-count", type=int)
    parser.add_argument("--trial-split-manifest", type=Path)
    parser.add_argument("--score-artifact-output", type=Path)
    parser.add_argument(
        "--trial-half", choices=("standard", "a", "b"), default="standard"
    )
    parser.add_argument("--expected-top5-count", type=int)
    args = parser.parse_args(argv)
    if args.trial_half == "standard" and args.trial_split_manifest is not None:
        parser.error("--trial-half standard rejects --trial-split-manifest")
    if args.trial_half in {"a", "b"} and args.trial_split_manifest is None:
        parser.error(f"--trial-half {args.trial_half} requires --trial-split-manifest")
    if args.enable_hungarian and not args.hungarian_output:
        parser.error("--enable-hungarian requires --hungarian-output")
    if args.enable_hungarian and not args.similarity_output:
        parser.error("--enable-hungarian requires --similarity-output")
    if args.hungarian_output and not args.enable_hungarian:
        parser.error("--hungarian-output requires --enable-hungarian")
    for name in ("expected_top1_count", "expected_top5_count"):
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def parse_time_slice(value: str) -> tuple[int, int]:
    pieces = [piece.strip() for piece in value.split(",")]
    if len(pieces) != 2:
        raise ValueError("--time-slice must contain exactly START,END")
    start, end = (int(piece) for piece in pieces)
    if start < 0 or end <= start:
        raise ValueError("--time-slice must satisfy 0 <= START < END")
    return start, end


def resolve_device_and_dtype(
    device_arg: str, dtype_arg: str
) -> tuple[torch.device, torch.dtype]:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map[dtype_arg]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU evaluation requires --dtype float32")
    return device, dtype


def build_dataset(args: argparse.Namespace) -> datasets.Dataset:
    trial_half = getattr(args, "trial_half", "standard")
    manifest = getattr(args, "trial_split_manifest", None)
    trial_indices = (
        None
        if trial_half == "standard"
        else load_trial_indices_by_image(manifest, trial_half)
    )
    brain_dataset = load_things_brain_dataset(
        data_directory=args.brain_directory,
        split="test",
        subject_ids=args.subject_id,
        brain_column="eeg",
        avg_trials=True,
        selected_channels=args.selected_channels,
        trial_indices_by_image=trial_indices,
        expected_trial_count=40 if trial_indices is not None else None,
    )
    image_dataset = load_image_dataset(
        dataset_name=args.dataset_name,
        image_directory=args.image_directory,
        split="test",
        cache_dir=args.cache_dir,
        image_column="image",
    ).select_columns(["image", "image_id"])
    dataset = merge_datasets_by_image_id(
        brain_dataset, image_dataset, is_main_process=True
    )
    dataset = dataset.cast_column("image", datasets.Image(decode=True))
    dataset.set_format("torch", columns="eeg", output_all_columns=True)
    return dataset


def make_collate_fn(
    processor: CLIPImageProcessor, start: int, end: int, subject_id: int
):
    def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
        brain = torch.stack([example["eeg"] for example in examples]).float()
        if end > brain.shape[-1]:
            raise ValueError(
                f"time slice end {end} exceeds EEG length {brain.shape[-1]}"
            )
        brain = brain[:, :, start:end].contiguous()
        images = [example["image"] for example in examples]
        pixel_values = processor(images, return_tensors="pt").pixel_values
        return {
            "brain_signals": brain,
            "pixel_values": pixel_values,
            "subject_ids": torch.full(
                (len(examples),), subject_id, dtype=torch.long
            ),
            "image_ids": [str(example["image_id"]) for example in examples],
        }

    return collate


def unique_index(values: list[str], label: str) -> dict[str, int]:
    index: dict[str, int] = {}
    for position, value in enumerate(values):
        if value in index:
            raise ValueError(f"duplicate {label} image_id: {value}")
        index[value] = position
    return index


def solve_one_to_one_assignment(similarity: torch.Tensor) -> torch.Tensor:
    """Maximize total similarity and return one gallery column per query row."""
    if similarity.ndim != 2:
        raise ValueError(
            f"Hungarian assignment requires a 2-D matrix, got {similarity.ndim}-D"
        )
    num_queries, gallery_size = similarity.shape
    if num_queries == 0 or num_queries != gallery_size:
        raise ValueError(
            "Hungarian assignment requires a non-empty square matrix, got "
            f"{tuple(similarity.shape)}"
        )
    if not bool(torch.isfinite(similarity).all()):
        raise ValueError("Hungarian assignment matrix contains NaN or Inf")

    # The solver sees similarity values only. Ground-truth IDs are used later,
    # solely to score the completed assignment.
    scores = np.ascontiguousarray(
        similarity.detach().cpu().numpy(), dtype=np.float64
    )
    row_indices, column_indices = linear_sum_assignment(scores, maximize=True)
    expected = np.arange(num_queries, dtype=np.int64)
    if not np.array_equal(np.sort(row_indices), expected):
        raise RuntimeError("Hungarian solver did not cover every query exactly once")
    if not np.array_equal(np.sort(column_indices), expected):
        raise RuntimeError("Hungarian solver did not use every gallery item exactly once")

    assigned_columns = np.full(num_queries, -1, dtype=np.int64)
    assigned_columns[row_indices] = column_indices
    if np.any(assigned_columns < 0):
        raise RuntimeError("Hungarian solver returned an incomplete assignment")
    return torch.from_numpy(assigned_columns)


def solve_assignment_with_order_permutations(
    similarity: torch.Tensor,
    row_permutation: np.ndarray,
    column_permutation: np.ndarray,
) -> torch.Tensor:
    """Solve after reordering rows/columns, then map columns to original order."""
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("permuted assignment requires a square similarity matrix")
    size = similarity.shape[0]
    expected = np.arange(size, dtype=np.int64)
    row_permutation = np.ascontiguousarray(row_permutation, dtype=np.int64)
    column_permutation = np.ascontiguousarray(
        column_permutation, dtype=np.int64
    )
    if row_permutation.shape != (size,) or not np.array_equal(
        np.sort(row_permutation), expected
    ):
        raise ValueError("row_permutation is not a complete permutation")
    if column_permutation.shape != (size,) or not np.array_equal(
        np.sort(column_permutation), expected
    ):
        raise ValueError("column_permutation is not a complete permutation")

    permuted_similarity = similarity[
        torch.from_numpy(row_permutation)
    ][:, torch.from_numpy(column_permutation)]
    assigned_permuted_columns = solve_one_to_one_assignment(
        permuted_similarity
    ).numpy()
    assigned_original_columns = np.full(size, -1, dtype=np.int64)
    assigned_original_columns[row_permutation] = column_permutation[
        assigned_permuted_columns
    ]
    return torch.from_numpy(assigned_original_columns)


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def write_similarity_bundle(
    path: Path,
    similarity: np.ndarray,
    query_ids: list[str],
    gallery_ids: list[str],
    targets: np.ndarray,
    solver_row_permutation: np.ndarray,
    solver_column_permutation: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cosine_similarity=similarity,
            query_ids=np.asarray(query_ids, dtype=np.str_),
            gallery_ids=np.asarray(gallery_ids, dtype=np.str_),
            target_gallery_indices=targets,
            solver_row_permutation=solver_row_permutation,
            solver_column_permutation=solver_column_permutation,
        )
    os.replace(temporary, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    start, end = parse_time_slice(args.time_slice)
    device, dtype = resolve_device_and_dtype(args.device, args.dtype)

    output_values = [args.metrics_output, args.predictions_output]
    if args.enable_hungarian:
        output_values.extend([args.hungarian_output, args.similarity_output])
    if args.score_artifact_output is not None:
        output_values.append(args.score_artifact_output)
    resolved_outputs = [str(Path(value).resolve()) for value in output_values]
    if len(resolved_outputs) != len(set(resolved_outputs)):
        raise ValueError("all evaluator output paths must be distinct")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    dataset = build_dataset(args)
    if len(dataset) != args.expected_num_samples:
        raise ValueError(
            f"expected {args.expected_num_samples} samples, found {len(dataset)}"
        )

    processor = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        local_files_only=args.local_files_only,
    )
    brain_model = BrainCLIPModel.from_pretrained(
        args.brain_model_path,
        local_files_only=args.local_files_only,
    ).to(device=device, dtype=dtype)
    vision_base = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        local_files_only=args.local_files_only,
    )
    vision_model = PeftModel.from_pretrained(
        vision_base,
        args.vision_adapter_path,
        is_trainable=False,
    ).to(device=device, dtype=dtype)
    brain_model.eval().requires_grad_(False)
    vision_model.eval().requires_grad_(False)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(processor, start, end, args.subject_id),
    )

    brain_chunks: list[torch.Tensor] = []
    image_chunks: list[torch.Tensor] = []
    query_ids: list[str] = []
    gallery_ids: list[str] = []
    with torch.inference_mode():
        for batch in dataloader:
            brain_signals = batch["brain_signals"].to(
                device=device, dtype=dtype, non_blocking=True
            )
            subject_ids = batch["subject_ids"].to(device=device, non_blocking=True)
            pixel_values = batch["pixel_values"].to(
                device=device, dtype=dtype, non_blocking=True
            )
            brain_features = brain_model.get_brain_features(
                brain_signals, subject_ids=subject_ids
            )
            image_features = vision_model(pixel_values)[0]
            brain_chunks.append(F.normalize(brain_features.float(), dim=-1).cpu())
            image_chunks.append(F.normalize(image_features.float(), dim=-1).cpu())
            ids = batch["image_ids"]
            query_ids.extend(ids)
            gallery_ids.extend(ids)

    brain_embeddings = torch.cat(brain_chunks, dim=0)
    image_embeddings = torch.cat(image_chunks, dim=0)
    if brain_embeddings.shape != image_embeddings.shape:
        raise ValueError(
            "brain/image embedding shape mismatch: "
            f"{tuple(brain_embeddings.shape)} vs {tuple(image_embeddings.shape)}"
        )
    if len(query_ids) != len(dataset) or len(gallery_ids) != len(dataset):
        raise ValueError("embedding/image_id count mismatch")

    query_index = unique_index(query_ids, "query")
    gallery_index = unique_index(gallery_ids, "gallery")
    if set(query_index) != set(gallery_index):
        missing = sorted(set(query_index) - set(gallery_index))[:5]
        extra = sorted(set(gallery_index) - set(query_index))[:5]
        raise ValueError(f"query/gallery ID mismatch; missing={missing}, extra={extra}")

    targets = torch.tensor([gallery_index[value] for value in query_ids])
    cosine_similarity = brain_embeddings @ image_embeddings.T
    logit_scale = float(brain_model.logit_scale.detach().float().exp().cpu())
    scaled_logits = cosine_similarity * logit_scale
    loss = float(F.cross_entropy(scaled_logits, targets).item())

    ranking = torch.argsort(cosine_similarity, dim=1, descending=True, stable=True)
    topk_indices = ranking[:, : min(5, len(gallery_ids))]
    topk_scores = cosine_similarity.gather(1, topk_indices)
    target_ranks = (
        ranking.eq(targets[:, None]).to(torch.int64).argmax(dim=1) + 1
    )
    correct_top1 = target_ranks.le(1)
    correct_top5 = target_ranks.le(5)
    top1_count = int(correct_top1.sum().item())
    top5_count = int(correct_top5.sum().item())
    sample_count = len(query_ids)
    similarity_array = np.ascontiguousarray(
        cosine_similarity.numpy(), dtype=np.float32
    )

    if (
        args.expected_top1_count is not None
        and top1_count != args.expected_top1_count
    ):
        raise RuntimeError(
            "standard Top-1 regression check failed: "
            f"expected {args.expected_top1_count}, found {top1_count}"
        )
    if (
        args.expected_top5_count is not None
        and top5_count != args.expected_top5_count
    ):
        raise RuntimeError(
            "standard Top-5 regression check failed: "
            f"expected {args.expected_top5_count}, found {top5_count}"
        )

    if args.score_artifact_output is not None:
        score_artifact = build_brainrw_score_artifact(
            similarity=similarity_array,
            query_embeddings=brain_embeddings.numpy(),
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            trial_half=args.trial_half,
            brain_model_path=Path(args.brain_model_path).resolve(),
            top1_count=top1_count,
            top5_count=top5_count,
        )
        write_score_artifact(args.score_artifact_output, score_artifact)

    hungarian_metrics: dict[str, Any] | None = None
    hungarian_path: Path | None = None
    similarity_path: Path | None = None
    if args.enable_hungarian:
        target_array = np.ascontiguousarray(targets.numpy(), dtype=np.int64)
        if similarity_array.shape != (sample_count, sample_count):
            raise ValueError(
                "global one-to-one evaluation requires a square full-set matrix; "
                f"found {similarity_array.shape}"
            )

        # Fix a row/column ordering that is independent of the ground truth.
        # This prevents the query/gallery diagonal order from acting as an
        # accidental tie-breaker when several assignments have equal objective.
        primary_order_rng = np.random.default_rng(
            HUNGARIAN_PRIMARY_ORDER_SEED
        )
        solver_row_permutation = np.ascontiguousarray(
            primary_order_rng.permutation(sample_count), dtype=np.int64
        )
        solver_column_permutation = np.ascontiguousarray(
            primary_order_rng.permutation(sample_count), dtype=np.int64
        )
        assigned_columns = solve_assignment_with_order_permutations(
            cosine_similarity,
            solver_row_permutation,
            solver_column_permutation,
        )
        repeated_columns = solve_assignment_with_order_permutations(
            cosine_similarity,
            solver_row_permutation,
            solver_column_permutation,
        )
        if not torch.equal(assigned_columns, repeated_columns):
            raise RuntimeError(
                "repeated Hungarian solves with fixed ordering returned "
                "different assignments"
            )

        assigned_array = np.ascontiguousarray(
            assigned_columns.numpy(), dtype=np.int64
        )
        raw_top1_array = np.ascontiguousarray(
            topk_indices[:, 0].numpy(), dtype=np.int64
        )
        if len(np.unique(assigned_array)) != sample_count:
            raise RuntimeError("Hungarian assignment is not one-to-one")

        row_indices = np.arange(sample_count, dtype=np.int64)
        assigned_scores = similarity_array[row_indices, assigned_array]
        raw_top1_scores = similarity_array[row_indices, raw_top1_array]
        row_max_scores = similarity_array.max(axis=1)
        if np.any(assigned_scores > row_max_scores + 1e-7):
            raise RuntimeError("assigned similarity exceeds its row maximum")

        # Audit ordering sensitivity without using accuracy to select a result.
        # The primary ordering above is fixed before any ground-truth scoring.
        ordering_audit_assignments: list[tuple[int, np.ndarray]] = [
            (HUNGARIAN_PRIMARY_ORDER_SEED, assigned_array.copy())
        ]
        ordering_audit_objectives: list[float] = [
            float(assigned_scores.sum(dtype=np.float64))
        ]
        for audit_seed in HUNGARIAN_AUDIT_ORDER_SEEDS:
            audit_rng = np.random.default_rng(audit_seed)
            audit_row_permutation = np.ascontiguousarray(
                audit_rng.permutation(sample_count), dtype=np.int64
            )
            audit_column_permutation = np.ascontiguousarray(
                audit_rng.permutation(sample_count), dtype=np.int64
            )
            audit_assignment = solve_assignment_with_order_permutations(
                cosine_similarity,
                audit_row_permutation,
                audit_column_permutation,
            ).numpy()
            audit_assignment = np.ascontiguousarray(
                audit_assignment, dtype=np.int64
            )
            ordering_audit_assignments.append(
                (audit_seed, audit_assignment)
            )
            ordering_audit_objectives.append(
                float(
                    similarity_array[row_indices, audit_assignment].sum(
                        dtype=np.float64
                    )
                )
            )
        if (
            max(ordering_audit_objectives)
            - min(ordering_audit_objectives)
            > 1e-10
        ):
            raise RuntimeError(
                "Hungarian objective changed under row/column permutation"
            )

        assignment_correct = assigned_columns.eq(targets)
        assignment_correct_count = int(assignment_correct.sum().item())
        ordering_audit_records: list[dict[str, Any]] = []
        for (audit_seed, audit_assignment), audit_objective in zip(
            ordering_audit_assignments,
            ordering_audit_objectives,
            strict=True,
        ):
            audit_correct_count = int(
                np.count_nonzero(audit_assignment == target_array)
            )
            ordering_audit_records.append(
                {
                    "order_seed": audit_seed,
                    "assignment_sha256": sha256_array(audit_assignment),
                    "objective_total_cosine_similarity": audit_objective,
                    "assignment_correct_count": audit_correct_count,
                    "assignment_percent": 100.0
                    * audit_correct_count
                    / sample_count,
                    "is_predeclared_primary": audit_seed
                    == HUNGARIAN_PRIMARY_ORDER_SEED,
                }
            )
        transitions: list[str] = []
        transition_query_ids: dict[str, list[str]] = {
            "correct_to_correct": [],
            "correct_to_wrong": [],
            "wrong_to_correct": [],
            "wrong_to_wrong": [],
        }
        for row_index, query_id in enumerate(query_ids):
            before = bool(correct_top1[row_index])
            after = bool(assignment_correct[row_index])
            if before and after:
                transition = "correct_to_correct"
            elif before and not after:
                transition = "correct_to_wrong"
            elif not before and after:
                transition = "wrong_to_correct"
            else:
                transition = "wrong_to_wrong"
            transitions.append(transition)
            transition_query_ids[transition].append(query_id)

        transition_counts = {
            name: len(ids) for name, ids in transition_query_ids.items()
        }
        if sum(transition_counts.values()) != sample_count:
            raise RuntimeError("Hungarian transition ledger is incomplete")
        if (
            transition_counts["correct_to_correct"]
            + transition_counts["correct_to_wrong"]
            != top1_count
        ):
            raise RuntimeError("transition ledger does not reproduce standard Top-1")
        expected_assignment_correct = (
            top1_count
            + transition_counts["wrong_to_correct"]
            - transition_counts["correct_to_wrong"]
        )
        if assignment_correct_count != expected_assignment_correct:
            raise RuntimeError("transition ledger does not reproduce assignment accuracy")

        raw_prediction_counts = np.bincount(
            raw_top1_array, minlength=sample_count
        )
        assigned_total = float(assigned_scores.sum(dtype=np.float64))
        raw_total = float(raw_top1_scores.sum(dtype=np.float64))
        assignment_fraction = assignment_correct_count / sample_count
        changed_assignment_count = int(
            np.count_nonzero(assigned_array != raw_top1_array)
        )

        similarity_path = Path(args.similarity_output).resolve()
        write_similarity_bundle(
            similarity_path,
            similarity_array,
            query_ids,
            gallery_ids,
            target_array,
            solver_row_permutation,
            solver_column_permutation,
        )

        hungarian_path = Path(args.hungarian_output).resolve()
        hungarian_path.parent.mkdir(parents=True, exist_ok=True)
        with hungarian_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "query_index",
                    "subject_id",
                    "query_image_id",
                    "gt_image_id",
                    "gt_gallery_index",
                    "gt_rank",
                    "gt_cosine_similarity",
                    "independent_top1_gallery_index",
                    "independent_top1_image_id",
                    "independent_top1_similarity",
                    "independent_top1_correct",
                    "assigned_gallery_index",
                    "assigned_image_id",
                    "assigned_similarity",
                    "assignment_correct",
                    "assignment_changed_from_independent_top1",
                    "transition",
                    "similarity_drop_from_independent_row_max",
                ],
            )
            writer.writeheader()
            for row_index, query_id in enumerate(query_ids):
                target_index = int(target_array[row_index])
                raw_index = int(raw_top1_array[row_index])
                assigned_index = int(assigned_array[row_index])
                writer.writerow(
                    {
                        "query_index": row_index,
                        "subject_id": args.subject_id,
                        "query_image_id": query_id,
                        "gt_image_id": query_id,
                        "gt_gallery_index": target_index,
                        "gt_rank": int(target_ranks[row_index]),
                        "gt_cosine_similarity": float(
                            similarity_array[row_index, target_index]
                        ),
                        "independent_top1_gallery_index": raw_index,
                        "independent_top1_image_id": gallery_ids[raw_index],
                        "independent_top1_similarity": float(
                            raw_top1_scores[row_index]
                        ),
                        "independent_top1_correct": int(correct_top1[row_index]),
                        "assigned_gallery_index": assigned_index,
                        "assigned_image_id": gallery_ids[assigned_index],
                        "assigned_similarity": float(assigned_scores[row_index]),
                        "assignment_correct": int(assignment_correct[row_index]),
                        "assignment_changed_from_independent_top1": int(
                            assigned_index != raw_index
                        ),
                        "transition": transitions[row_index],
                        "similarity_drop_from_independent_row_max": float(
                            row_max_scores[row_index] - assigned_scores[row_index]
                        ),
                    }
                )

        hungarian_metrics = {
            "metric_name": "one_to_one_constrained_assignment_accuracy",
            "protocol": "global_one_to_one_assignment",
            "evaluation_scope": "transductive_closed_set_full_test_batch",
            "method": "scipy.optimize.linear_sum_assignment",
            "solver_version": scipy.__version__,
            "objective": "maximize_total_cosine_similarity",
            "solver_input": "cosine_similarity_matrix_only",
            "ground_truth_used_only_after_assignment": True,
            "one_to_one_constraint": True,
            "joint_batch": True,
            "matrix_shape": list(similarity_array.shape),
            "matrix_dtype": str(similarity_array.dtype),
            "matrix_sha256": sha256_array(similarity_array),
            "query_ids_sha256": sha256_strings(query_ids),
            "gallery_ids_sha256": sha256_strings(gallery_ids),
            "targets_sha256": sha256_array(target_array),
            "tie_break_policy": (
                "predeclared_seeded_independent_row_and_column_permutations"
            ),
            "primary_order_seed": HUNGARIAN_PRIMARY_ORDER_SEED,
            "primary_selected_before_ground_truth_scoring": True,
            "solver_row_permutation_sha256": sha256_array(
                solver_row_permutation
            ),
            "solver_column_permutation_sha256": sha256_array(
                solver_column_permutation
            ),
            "query_unique_count": len(query_index),
            "gallery_unique_count": len(gallery_index),
            "query_gallery_id_sets_equal": set(query_index) == set(gallery_index),
            "assigned_gallery_unique_count": int(len(np.unique(assigned_array))),
            "assignment_sha256": sha256_array(assigned_array),
            "repeat_solver_assignment_equal": True,
            "ordering_sensitivity_audit": {
                "order_seeds": [
                    HUNGARIAN_PRIMARY_ORDER_SEED,
                    *HUNGARIAN_AUDIT_ORDER_SEEDS,
                ],
                "run_count": len(ordering_audit_records),
                "unique_assignment_count": len(
                    {
                        record["assignment_sha256"]
                        for record in ordering_audit_records
                    }
                ),
                "all_mapped_assignments_equal": len(
                    {
                        record["assignment_sha256"]
                        for record in ordering_audit_records
                    }
                )
                == 1,
                "objective_min": min(ordering_audit_objectives),
                "objective_max": max(ordering_audit_objectives),
                "objective_spread": max(ordering_audit_objectives)
                - min(ordering_audit_objectives),
                "assignment_correct_count_min": min(
                    record["assignment_correct_count"]
                    for record in ordering_audit_records
                ),
                "assignment_correct_count_max": max(
                    record["assignment_correct_count"]
                    for record in ordering_audit_records
                ),
                "accuracy_not_used_for_selection": True,
                "runs": ordering_audit_records,
            },
            "assignment_correct_count": assignment_correct_count,
            "assignment_accuracy": assignment_fraction,
            "assignment_fraction": assignment_fraction,
            "assignment_percent": 100.0 * assignment_fraction,
            "delta_count_vs_independent_top1": assignment_correct_count
            - top1_count,
            "delta_percentage_points_vs_independent_top1": 100.0
            * (assignment_fraction - top1_count / sample_count),
            "changed_assignment_count": changed_assignment_count,
            "total_assigned_cosine_similarity": assigned_total,
            "mean_assigned_cosine_similarity": assigned_total / sample_count,
            "independent_rowwise_max_total_cosine_similarity": raw_total,
            "one_to_one_constraint_similarity_cost": raw_total - assigned_total,
            "independent_top1_unique_gallery_count": int(
                np.count_nonzero(raw_prediction_counts)
            ),
            "independent_top1_duplicate_excess": int(
                sample_count - np.count_nonzero(raw_prediction_counts)
            ),
            "independent_top1_collided_gallery_count": int(
                np.count_nonzero(raw_prediction_counts > 1)
            ),
            "independent_top1_max_collision_multiplicity": int(
                raw_prediction_counts.max()
            ),
            "transitions": {
                name: {"count": len(ids), "query_image_ids": ids}
                for name, ids in transition_query_ids.items()
            },
            "standard_reference": {
                "independent_top1_count": top1_count,
                "independent_top1_percent": 100.0 * top1_count / sample_count,
                "independent_top5_count": top5_count,
                "independent_top5_percent": 100.0 * top5_count / sample_count,
            },
            "top5_status": "not_defined_for_single_global_assignment",
            "extra_assumption": (
                "The complete query set and gallery form a known bijection; "
                "each gallery image must be assigned exactly once."
            ),
        }

    predictions_path = Path(args.predictions_output).resolve()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_index",
                "subject_id",
                "gt_image_id",
                "gt_rank",
                "gt_cosine_similarity",
                "top1_image_id",
                "top1_cosine_similarity",
                "top5_image_ids",
                "top5_cosine_similarities",
                "correct_top1",
                "correct_top5",
            ],
        )
        writer.writeheader()
        for row_index, image_id in enumerate(query_ids):
            indices = topk_indices[row_index].tolist()
            scores = [float(value) for value in topk_scores[row_index].tolist()]
            target_index = int(targets[row_index])
            writer.writerow(
                {
                    "query_index": row_index,
                    "subject_id": args.subject_id,
                    "gt_image_id": image_id,
                    "gt_rank": int(target_ranks[row_index]),
                    "gt_cosine_similarity": float(
                        cosine_similarity[row_index, target_index]
                    ),
                    "top1_image_id": gallery_ids[indices[0]],
                    "top1_cosine_similarity": scores[0],
                    "top5_image_ids": json.dumps(
                        [gallery_ids[index] for index in indices]
                    ),
                    "top5_cosine_similarities": json.dumps(scores),
                    "correct_top1": int(correct_top1[row_index]),
                    "correct_top5": int(correct_top5[row_index]),
                }
            )

    top1 = top1_count / sample_count
    top5 = top5_count / sample_count
    metrics = {
        "schema_version": 1,
        "task": "THINGS-EEG brain-to-image retrieval",
        "subject_id": args.subject_id,
        "seed": args.seed,
        "split": "test",
        "checkpoint_policy": "final_epoch",
        "sample_count": sample_count,
        "gallery_size": len(gallery_ids),
        "embedding_dim": int(brain_embeddings.shape[1]),
        "top1_count": top1_count,
        "top5_count": top5_count,
        "top1": top1,
        "top5": top5,
        "top1_fraction": top1,
        "top5_fraction": top5,
        "top1_percent": 100.0 * top1,
        "top5_percent": 100.0 * top5,
        "loss": loss,
        "mean_gt_cosine_similarity": float(
            cosine_similarity[torch.arange(sample_count), targets].mean().item()
        ),
        "logit_scale": logit_scale,
        "chance_top1": 1.0 / len(gallery_ids),
        "chance_top5": min(5, len(gallery_ids)) / len(gallery_ids),
        "protocol": {
            "trial_averaging": True,
            "time_slice": [start, end],
            "selected_channels": [
                value.strip()
                for value in args.selected_channels.split(",")
                if value.strip()
            ],
            "similarity": "cosine",
            "target_match": "unique image_id",
        },
        "paths": {
            "brain_model": str(Path(args.brain_model_path).resolve()),
            "vision_adapter": str(Path(args.vision_adapter_path).resolve()),
            "clip_base": str(Path(args.pretrained_model_name_or_path).resolve()),
            "brain_directory": str(Path(args.brain_directory).resolve()),
            "image_directory": str(Path(args.image_directory).resolve()),
            "predictions": str(predictions_path),
        },
        "environment": {
            "python": platform.python_version(),
            "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "peft": peft.__version__,
            "device": str(device),
            "dtype": args.dtype,
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    if hungarian_metrics is not None:
        metrics["hungarian_assignment"] = hungarian_metrics
        if hungarian_path is None or similarity_path is None:
            raise RuntimeError("Hungarian output paths were not initialized")
        metrics["paths"]["hungarian_assignment_predictions"] = str(
            hungarian_path
        )
        metrics["paths"]["cosine_similarity_matrix"] = str(similarity_path)
    metrics_path = Path(args.metrics_output).resolve()
    write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
