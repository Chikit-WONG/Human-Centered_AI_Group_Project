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
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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


def _git_provenance() -> dict[str, object]:
    root = _REPOSITORY_ROOT

    def output(*arguments: str) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *arguments],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (
            OSError,
            subprocess.CalledProcessError,
        ) as exc:
            raise RuntimeError(
                "Git provenance cannot be resolved"
            ) from exc

    actual_root = output(
        "rev-parse",
        "--show-toplevel",
    )
    try:
        resolved_actual = Path(actual_root).resolve(strict=True)
        resolved_expected = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            "Git repository root cannot be resolved"
        ) from exc
    if resolved_actual != resolved_expected:
        raise RuntimeError(
            "Git repository root differs from the anchored project"
        )
    revision = output("rev-parse", "HEAD")
    if _GIT_SHA_RE.fullmatch(revision) is None:
        raise ValueError("Git SHA is invalid")
    status = output(
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if status:
        raise RuntimeError(
            "Git worktree must be clean before BrainRW execution"
        )
    return {
        "clean": True,
        "git_sha": revision,
        "repository_root": str(resolved_expected),
    }


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


def _validate_resume_model_before_data(
    payload: Mapping[str, object],
    *,
    model: br.BrainRWCLIPLoRAModel,
    effective_batch_size: int,
    task_initialization_sha256: str,
    candidate_initialization_sha256: str,
) -> None:
    expected = {
        "model_manifest_sha256": model.model_manifest_sha256,
        "effective_batch_size": effective_batch_size,
        "task_initialization_sha256": task_initialization_sha256,
        "candidate_initialization_sha256": (
            candidate_initialization_sha256
        ),
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"resume checkpoint {name} mismatch")


def _load_resume_runtime_before_data(
    payload: Mapping[str, object],
    *,
    model: br.BrainRWCLIPLoRAModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    sampler: br.StatefulIndexSampler,
    loader_generator: torch.Generator,
    device: torch.device,
    planned_steps: int,
    global_step: int,
) -> None:
    try:
        model.load_checkpoint_states(
            payload["task_state"],
            payload["candidate_state"],
        )
        optimizer_state = payload["optimizer_state"]
        if not isinstance(optimizer_state, Mapping):
            raise ValueError(
                "optimizer state must be a mapping"
            )
        expected_optimizer = optimizer.state_dict()
        if set(optimizer_state) != set(expected_optimizer):
            raise ValueError(
                "optimizer state has an unexpected schema"
            )
        actual_groups = optimizer_state.get("param_groups")
        expected_groups = expected_optimizer["param_groups"]
        if (
            not isinstance(actual_groups, list)
            or len(actual_groups) != len(expected_groups)
        ):
            raise ValueError(
                "optimizer parameter groups differ from recipe"
            )
        parameter_by_id: dict[int, torch.nn.Parameter] = {}
        for actual, expected, runtime_group in zip(
            actual_groups,
            expected_groups,
            optimizer.param_groups,
            strict=True,
        ):
            if (
                not isinstance(actual, Mapping)
                or set(actual) != set(expected)
                or not isinstance(actual.get("params"), list)
                or len(actual["params"]) != len(expected["params"])
            ):
                raise ValueError(
                    "optimizer parameter-group schema differs from recipe"
                )
            parameter_ids = actual["params"]
            runtime_parameters = runtime_group["params"]
            if len(parameter_ids) != len(runtime_parameters):
                raise ValueError(
                    "optimizer parameter count differs from recipe"
                )
            for parameter_id, parameter in zip(
                parameter_ids,
                runtime_parameters,
                strict=True,
            ):
                if (
                    type(parameter_id) is not int
                    or parameter_id < 0
                    or parameter_id in parameter_by_id
                    or not isinstance(
                        parameter,
                        torch.nn.Parameter,
                    )
                ):
                    raise ValueError(
                        "optimizer parameter identity is invalid"
                    )
                parameter_by_id[parameter_id] = parameter
        optimizer_values = optimizer_state.get("state")
        if (
            not isinstance(optimizer_values, Mapping)
            or set(optimizer_values) != set(parameter_by_id)
        ):
            raise ValueError(
                "optimizer state does not cover every trainable parameter"
            )
        for parameter_id, parameter in parameter_by_id.items():
            state = optimizer_values[parameter_id]
            if (
                not isinstance(state, Mapping)
                or set(state)
                != {"step", "exp_avg", "exp_avg_sq"}
            ):
                raise ValueError(
                    "optimizer parameter state has an unexpected schema"
                )
            step = state["step"]
            if (
                not isinstance(step, torch.Tensor)
                or step.numel() != 1
                or not step.is_floating_point()
                or not bool(torch.isfinite(step).all())
                or float(step.item()) != float(global_step)
            ):
                raise ValueError(
                    "optimizer parameter step differs from global step"
                )
            for name in ("exp_avg", "exp_avg_sq"):
                moment = state[name]
                if (
                    not isinstance(moment, torch.Tensor)
                    or moment.shape != parameter.shape
                    or not moment.is_floating_point()
                    or moment.dtype
                    not in {parameter.dtype, torch.float32}
                    or not bool(torch.isfinite(moment).all())
                    or (
                        name == "exp_avg_sq"
                        and bool((moment < 0).any())
                    )
                ):
                    raise ValueError(
                        f"optimizer {name} tensor is invalid"
                    )
        optimizer.load_state_dict(optimizer_state)
        if [
            group.get("group_name")
            for group in optimizer.param_groups
        ] != ["brain_task", "clip_lora"]:
            raise ValueError(
                "optimizer parameter-group identity mismatch"
            )
        _move_optimizer_state(optimizer, device)

        scheduler_state = payload["scheduler_state"]
        if (
            not isinstance(scheduler_state, Mapping)
            or set(scheduler_state)
            != set(scheduler.state_dict())
        ):
            raise ValueError(
                "scheduler state has an unexpected schema"
            )
        if scheduler_state.get("last_epoch") != global_step:
            raise ValueError(
                "scheduler step differs from global step"
            )
        scheduler.load_state_dict(scheduler_state)
        if (
            scheduler.state_dict().get("T_max") != planned_steps
            or scheduler.last_epoch != global_step
        ):
            raise ValueError(
                "scheduler progress differs from planned steps"
            )
        sampler.load_state_dict(payload["sampler_state"])
        loader_generator.set_state(
            payload["dataloader_generator_state"]
        )
        br.restore_rng_state(payload["rng_state"])
    except (
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"resume checkpoint runtime state is invalid: {exc}"
        ) from exc


def _validation_metrics(
    model: br.BrainRWCLIPLoRAModel,
    dataset: object,
    processor: object,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    similarity, identifiers = br.evaluate_brainrw_similarity(
        model,
        dataset,
        processor,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
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
    initial_git_provenance = _git_provenance()
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
        br.validate_brainrw_checkpoint_identity(
            resume.payload,
            config=config,
            manifest=manifest,
            subject=args.subject,
            seed=args.seed,
        )
        if resume.payload["git_sha"] != initial_git_provenance[
            "git_sha"
        ]:
            raise ValueError(
                "resume checkpoint Git revision mismatch"
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, processor = br.build_brainrw_model(
        config.payload,
        config.clip_path,
        expected_preprocessor_sha256=(
            config.clip_preprocessor_sha256
        ),
    )
    training = config.payload["training"]
    optimizer_config = config.payload["optimizer"]
    assert isinstance(training, Mapping)
    assert isinstance(optimizer_config, Mapping)
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    effective_batch_size = batch_size
    task_initialization_sha256 = br.state_dict_sha256(
        model.task_state_dict()
    )
    candidate_initialization_sha256 = br.state_dict_sha256(
        model.candidate_state_dict()
    )
    if resume is not None:
        _validate_resume_model_before_data(
            resume.payload,
            model=model,
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
    runtime_dtype_name = (
        "bfloat16"
        if runtime_dtype is torch.bfloat16
        else "float32"
    )
    if resume is not None and br.checkpoint_runtime_dtype(
        resume.payload,
        device,
    ) is not runtime_dtype:
        raise ValueError("resume checkpoint runtime_dtype mismatch")
    model.to(device=device, dtype=runtime_dtype)
    optimizer = _build_optimizer(model, optimizer_config)
    sampler = br.StatefulIndexSampler(
        manifest.train_row_count,
        args.seed,
    )
    planned_steps = epochs * max(
        1, math.ceil(sampler.size / batch_size)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=planned_steps,
    )
    loader_generator = torch.Generator().manual_seed(
        args.seed + 1_000_003
    )
    global_step = 0
    resumed_from_sha256 = None
    if resume is not None:
        payload = resume.payload
        if int(payload["planned_steps"]) != planned_steps:
            raise ValueError("resume planned optimization steps mismatch")
        _load_resume_runtime_before_data(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
            loader_generator=loader_generator,
            device=device,
            planned_steps=planned_steps,
            global_step=int(payload["global_step"]),
        )
        global_step = int(payload["global_step"])
        resumed_from_sha256 = resume.sha256

    train_dataset = br.BrainRWDevelopmentDataset(
        manifest.path,
        "train",
        args.seed,
        expected_source_payload_sha256=manifest.source_payload_sha256,
    )
    val_dataset = br.BrainRWDevelopmentDataset(
        manifest.path,
        "val-dev",
        args.seed,
        expected_source_payload_sha256=manifest.source_payload_sha256,
    )
    if (
        train_dataset.subject_id != args.subject
        or val_dataset.subject_id != args.subject
    ):
        raise ValueError("dataset subject differs from CLI subject")
    if (
        len(train_dataset) != sampler.size
        or len(val_dataset) != manifest.val_dev_row_count
    ):
        raise ValueError(
            "dataset row count differs from manifest identity"
        )
    base_data_order_sha256 = br.data_order_sha256(
        train_dataset,
        sampler,
    )
    if (
        resume is not None
        and resume.payload["data_order_sha256"]
        != base_data_order_sha256
    ):
        raise ValueError(
            "resume checkpoint data_order_sha256 mismatch"
        )

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
        dtype=runtime_dtype,
    )
    final_git_provenance = _git_provenance()
    if final_git_provenance != initial_git_provenance:
        raise RuntimeError(
            "Git provenance changed during BrainRW execution"
        )
    git_sha = str(initial_git_provenance["git_sha"])
    training_complete = global_step == planned_steps
    payload = {
        "schema_version": 1,
        "payload_type": br.BRAINRW_CHECKPOINT_TYPE,
        "complete": True,
        "training_complete": training_complete,
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
        "clip_preprocessor_sha256": config.clip_preprocessor_sha256,
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
        "runtime_dtype": runtime_dtype_name,
        "environment": br.capture_environment(),
        "git_sha": git_sha,
        "git_provenance": _json_clone(
            initial_git_provenance
        ),
        "validation_metrics": metrics,
        "resumed_from_sha256": resumed_from_sha256,
    }
    br.validate_brainrw_checkpoint_identity(
        payload,
        config=config,
        manifest=manifest,
        subject=args.subject,
        seed=args.seed,
    )
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
        "training_complete": training_complete,
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
        "runtime_dtype": runtime_dtype_name,
        "git_sha": git_sha,
        "git_provenance": initial_git_provenance,
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
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
