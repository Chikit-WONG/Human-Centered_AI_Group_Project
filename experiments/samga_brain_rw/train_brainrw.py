#!/usr/bin/env python3
"""Train the sealed development-only Brain-RW/CLIP-LoRA branch."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from samga_brain_rw import brainrw as br
from samga_brain_rw.hashing import canonical_json_bytes
from samga_brain_rw.scores import independent_retrieval_metrics


_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", required=True, choices=["train"])
    parser.add_argument(
        "--validation-scope",
        required=True,
        choices=["val-dev"],
    )
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--resume",
        required=True,
        help="Literal 'none' or an explicit typed checkpoint path",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--clip-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-train-steps", type=int)
    return parser.parse_args(argv)


def _git_sha() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise RuntimeError(
            "Git SHA cannot be resolved"
        ) from exc
    if _GIT_SHA_RE.fullmatch(value) is None:
        raise ValueError("Git SHA is invalid")
    return value


def _json_clone(value: object) -> object:
    return json.loads(canonical_json_bytes(value))


def _validate_cli(args: argparse.Namespace) -> None:
    if not 1 <= args.subject <= 10:
        raise ValueError("subject must be between 1 and 10")
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.resume != "none" and not args.resume.strip():
        raise ValueError("resume must be literal 'none' or a checkpoint path")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("max-train-steps must be positive")
    output = br.reject_development_path(args.output_dir, "output directory")
    if output.exists():
        raise FileExistsError("output directory already exists")
    if not output.parent.is_dir():
        raise ValueError("output directory parent must already exist")


def _build_optimizer(
    model: br.BrainRWCLIPLoRAModel,
    optimizer_config: Mapping[str, object],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {
                "params": model.brain_parameters(),
                "lr": float(optimizer_config["brain_learning_rate"]),
                "group_name": "brain_task",
            },
            {
                "params": model.lora_parameters(),
                "lr": float(optimizer_config["visual_learning_rate"]),
                "group_name": "clip_lora",
            },
        ],
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _validate_resume_before_data(
    payload: Mapping[str, object],
    *,
    subject: int,
    seed: int,
    config: br.BrainRWConfigIdentity,
    manifest: br.ManifestIdentity,
    run_key: str,
    input_hashes: Mapping[str, str],
) -> None:
    comparisons = {
        "subject": subject,
        "seed": seed,
        "config_sha256": config.sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "run_key": run_key,
        "clip_config_sha256": config.clip_config_sha256,
        "clip_weights_sha256": config.clip_weights_sha256,
    }
    for name, expected in comparisons.items():
        if payload.get(name) != expected:
            raise ValueError(f"resume checkpoint {name} mismatch")
    if payload.get("input_hashes") != dict(input_hashes):
        raise ValueError("resume checkpoint input-hash bundle mismatch")


def _validate_resume_after_model(
    payload: Mapping[str, object],
    *,
    model: br.BrainRWCLIPLoRAModel,
    data_order_hash: str,
    effective_batch_size: int,
    task_initialization_sha256: str,
    candidate_initialization_sha256: str,
) -> None:
    expected = {
        "model_manifest_sha256": model.model_manifest_sha256,
        "data_order_sha256": data_order_hash,
        "effective_batch_size": effective_batch_size,
        "task_initialization_sha256": task_initialization_sha256,
        "candidate_initialization_sha256": (
            candidate_initialization_sha256
        ),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"resume checkpoint {name} mismatch")


def _validation_metrics(
    model: br.BrainRWCLIPLoRAModel,
    dataset: object,
    processor: object,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    similarity, identifiers = br.evaluate_brainrw_similarity(
        model,
        dataset,
        processor,
        batch_size=batch_size,
        device=device,
    )
    metrics = independent_retrieval_metrics(
        similarity,
        identifiers,
        identifiers,
    )
    return {
        "query_count": metrics.query_count,
        "gallery_count": metrics.gallery_count,
        "top1_count": metrics.top1_count,
        "top5_count": metrics.top5_count,
        "top1_rate": metrics.top1_rate,
        "top5_rate": metrics.top5_rate,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_cli(args)
    config = br.verify_brainrw_config(args.config, args.clip_path)
    manifest = br.load_development_manifest_identity(
        args.manifest,
        expected_subject=args.subject,
    )
    run_key, input_bundle_sha256, hashes = br.brainrw_run_key(
        config,
        manifest,
        args.subject,
        args.seed,
    )
    resume = None
    if args.resume != "none":
        resume = br.load_brainrw_checkpoint(
            Path(args.resume),
            requested_scope="train",
        )
        _validate_resume_before_data(
            resume.payload,
            subject=args.subject,
            seed=args.seed,
            config=config,
            manifest=manifest,
            run_key=run_key,
            input_hashes=hashes,
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_dataset = br.BrainRWDevelopmentDataset(
        manifest.path,
        "train",
        args.seed,
    )
    val_dataset = br.BrainRWDevelopmentDataset(
        manifest.path,
        "val-dev",
        args.seed,
    )
    if (
        train_dataset.subject_id != args.subject
        or val_dataset.subject_id != args.subject
    ):
        raise ValueError("dataset subject differs from CLI subject")

    model, processor = br.build_brainrw_model(
        config.payload,
        config.clip_path,
    )
    training = config.payload["training"]
    optimizer_config = config.payload["optimizer"]
    assert isinstance(training, Mapping)
    assert isinstance(optimizer_config, Mapping)
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    effective_batch_size = batch_size
    sampler = br.StatefulIndexSampler(
        len(train_dataset),
        args.seed,
    )
    base_data_order_sha256 = br.data_order_sha256(
        train_dataset,
        sampler,
    )
    loader_generator = torch.Generator().manual_seed(
        args.seed + 1_000_003
    )
    task_initialization_sha256 = br.state_dict_sha256(
        model.task_state_dict()
    )
    candidate_initialization_sha256 = br.state_dict_sha256(
        model.candidate_state_dict()
    )
    if resume is not None:
        _validate_resume_after_model(
            resume.payload,
            model=model,
            data_order_hash=base_data_order_sha256,
            effective_batch_size=effective_batch_size,
            task_initialization_sha256=task_initialization_sha256,
            candidate_initialization_sha256=(
                candidate_initialization_sha256
            ),
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    runtime_dtype = (
        torch.bfloat16
        if device.type == "cuda"
        and training["precision"] == "bf16"
        else torch.float32
    )
    model.to(device=device, dtype=runtime_dtype)
    optimizer = _build_optimizer(model, optimizer_config)
    planned_steps = epochs * max(
        1, math.ceil(len(train_dataset) / batch_size)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=planned_steps,
    )
    global_step = 0
    resumed_from_sha256 = None
    if resume is not None:
        payload = resume.payload
        model.load_checkpoint_states(
            payload["task_state"],
            payload["candidate_state"],
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer_state(optimizer, device)
        scheduler.load_state_dict(payload["scheduler_state"])
        sampler.load_state_dict(payload["sampler_state"])
        loader_generator.set_state(
            payload["dataloader_generator_state"]
        )
        global_step = int(payload["global_step"])
        if int(payload["planned_steps"]) != planned_steps:
            raise ValueError("resume planned optimization steps mismatch")
        br.restore_rng_state(payload["rng_state"])
        resumed_from_sha256 = resume.sha256

    stop_step = (
        planned_steps
        if args.max_train_steps is None
        else min(planned_steps, args.max_train_steps)
    )
    if global_step >= stop_step:
        raise ValueError(
            "resume checkpoint already reached max-train-steps"
        )
    model.train()
    while global_step < stop_step and sampler.epoch < epochs:
        if sampler.position == sampler.size:
            sampler.advance_epoch()
            if sampler.epoch >= epochs:
                break
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
            generator=loader_generator,
            collate_fn=br.BrainRWCollator(processor),
        )
        produced_batch = False
        for batch in loader:
            produced_batch = True
            optimizer.zero_grad(set_to_none=True)
            output = model(
                brain_signals=batch["brain_signals"].to(
                    device=device,
                    dtype=runtime_dtype,
                ),
                pixel_values=batch["pixel_values"].to(
                    device=device,
                    dtype=runtime_dtype,
                ),
            )
            if output.loss is None:
                raise AssertionError("training model did not return loss")
            output.loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step >= stop_step:
                break
        if not produced_batch:
            raise ValueError("training loader produced no batches")

    metrics = _validation_metrics(
        model,
        val_dataset,
        processor,
        batch_size=batch_size,
        device=device,
    )
    git_sha = _git_sha()
    payload = {
        "schema_version": 1,
        "payload_type": br.BRAINRW_CHECKPOINT_TYPE,
        "complete": True,
        "scope": "train",
        "validation_scope": "val-dev",
        "observed_scopes": ["train", "val-dev"],
        "subject": args.subject,
        "seed": args.seed,
        "config_path": str(config.path),
        "config_payload": _json_clone(dict(config.payload)),
        "config_sha256": config.sha256,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "input_hashes": hashes,
        "input_bundle_sha256": input_bundle_sha256,
        "run_key": run_key,
        "clip_path": str(config.clip_path),
        "clip_config_sha256": config.clip_config_sha256,
        "clip_weights_sha256": config.clip_weights_sha256,
        "model_manifest": _json_clone(dict(model.model_manifest)),
        "model_manifest_sha256": model.model_manifest_sha256,
        "target_manifest_sha256": model.target_manifest_sha256,
        "task_state": model.task_state_dict(),
        "candidate_state": model.candidate_state_dict(),
        "task_initialization_sha256": task_initialization_sha256,
        "candidate_initialization_sha256": (
            candidate_initialization_sha256
        ),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": sampler.epoch,
        "global_step": global_step,
        "planned_steps": planned_steps,
        "rng_state": br.capture_rng_state(),
        "sampler_state": sampler.state_dict(),
        "dataloader_generator_state": (
            loader_generator.get_state()
        ),
        "data_order_sha256": base_data_order_sha256,
        "effective_batch_size": effective_batch_size,
        "steps": global_step,
        "environment": br.capture_environment(),
        "git_sha": git_sha,
        "validation_metrics": metrics,
        "resumed_from_sha256": resumed_from_sha256,
    }
    output = br.reject_development_path(
        args.output_dir, "output directory"
    )
    br.create_development_directory_exclusive(
        output,
        context="output directory",
    )
    checkpoint_hash = br.save_brainrw_checkpoint(
        output / "checkpoint.pt",
        payload,
        manifest,
    )
    run_manifest = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.brainrw_run_manifest",
        "complete": True,
        "scope": "train",
        "validation_scope": "val-dev",
        "observed_scopes": ["train", "val-dev"],
        "subject": args.subject,
        "seed": args.seed,
        "run_key": run_key,
        "config_sha256": config.sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "protocol_sha256": manifest.protocol_sha256,
        "input_hashes": hashes,
        "input_bundle_sha256": input_bundle_sha256,
        "checkpoint_sha256": checkpoint_hash,
        "model_manifest_sha256": model.model_manifest_sha256,
        "target_manifest_sha256": model.target_manifest_sha256,
        "data_order_sha256": base_data_order_sha256,
        "task_initialization_sha256": task_initialization_sha256,
        "candidate_initialization_sha256": (
            candidate_initialization_sha256
        ),
        "effective_batch_size": effective_batch_size,
        "planned_steps": planned_steps,
        "completed_steps": global_step,
        "validation_metrics": metrics,
        "resumed_from_sha256": resumed_from_sha256,
    }
    br.write_development_file_exclusive(
        output / "run_manifest.json",
        canonical_json_bytes(run_manifest) + b"\n",
        context="run manifest output",
    )
    return run_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    try:
        args = parse_args(argv)
        run(args)
    except SystemExit:
        raise
    except (
        FileExistsError,
        OSError,
        PermissionError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
