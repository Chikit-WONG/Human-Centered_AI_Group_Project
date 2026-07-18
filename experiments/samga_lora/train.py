#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import get_peft_model_state_dict
from torch.utils.data import DataLoader

EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import SAMGACollator, ThingsEEGSubjectDataset, load_manifest  # noqa: E402
from samga_lora.model import SAMGALoss, SAMGATaskModel, load_clip_provider  # noqa: E402
from samga_lora.utils import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    git_revision,
    hash_file,
    hash_state_dict,
    parameter_count,
    retrieval_metrics,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAMGA with frozen CLIP or visual LoRA")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset", choices=["pilot_train", "formal_train"], required=True)
    parser.add_argument("--vision-mode", choices=["frozen", "lora"], required=True)
    parser.add_argument("--feature-cache", default=None)
    parser.add_argument("--clip-path", required=True)
    parser.add_argument("--subject-id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--prior-center", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--vision-lr-ratio", type=float, default=0.1)
    parser.add_argument("--stage1-lr", type=float, default=1e-4)
    parser.add_argument("--stage2-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stage1-epochs", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=60)
    parser.add_argument("--candidate-epochs", default="20,25,30,35,40,45,50,55,60")
    parser.add_argument("--mmd-start", type=float, default=0.9)
    parser.add_argument("--mmd-end", type=float, default=0.5)
    parser.add_argument("--eeg-l2norm", action="store_true", default=False)
    parser.add_argument(
        "--image-l2norm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--validation-manifest", default=None)
    parser.add_argument("--validation-feature-cache", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mmd_weight(epoch: int, stage1_epochs: int, start: float, end: float) -> float:
    if epoch > stage1_epochs:
        return 0.0
    if stage1_epochs <= 1:
        return float(end)
    progress = (epoch - 1) / (stage1_epochs - 1)
    return float(start + progress * (end - start))


def make_optimizer(
    task: SAMGATaskModel,
    vision_provider: torch.nn.Module | None,
    *,
    task_lr: float,
    vision_lr: float,
    weight_decay: float,
    include_shared: bool,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = [
        {"params": task.task_parameters(include_shared=include_shared), "lr": task_lr, "name": "task"}
    ]
    if vision_provider is not None:
        vision_parameters = [parameter for parameter in vision_provider.parameters() if parameter.requires_grad]
        if vision_parameters:
            groups.append({"params": vision_parameters, "lr": vision_lr, "name": "vision_lora"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=weight_decay)


def gradient_norm(parameters: Any) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().float().pow(2).sum())
    return squared ** 0.5


def vision_features(
    batch: dict[str, Any],
    vision_provider: torch.nn.Module | None,
    device: torch.device,
) -> torch.Tensor:
    if "layer_features" in batch:
        return batch["layer_features"].to(device=device, dtype=torch.float32, non_blocking=True)
    if vision_provider is None:
        raise RuntimeError("Online image batch received without a vision provider")
    pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return vision_provider(pixel_values).float()


@torch.no_grad()
def validate(
    task: SAMGATaskModel,
    vision_provider: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    task.eval()
    if vision_provider is not None:
        vision_provider.eval()
    all_eeg: list[torch.Tensor] = []
    all_image: list[torch.Tensor] = []
    image_ids: list[str] = []
    for batch in loader:
        eeg = batch["eeg"].to(device=device, dtype=torch.float32, non_blocking=True)
        subject_ids = batch["subject_ids"].to(device=device, non_blocking=True)
        layers = vision_features(batch, vision_provider, device)
        eeg_features, image_features, _ = task(eeg, layers, subject_ids, force_global=True)
        all_eeg.append(eeg_features.cpu())
        all_image.append(image_features.cpu())
        image_ids.extend(batch["image_ids"])
    metrics, _ = retrieval_metrics(
        torch.cat(all_eeg), torch.cat(all_image), image_ids, image_ids
    )
    return metrics


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    task: SAMGATaskModel,
    vision_provider: torch.nn.Module | None,
    config: dict[str, Any],
    metrics: dict[str, Any] | None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "epoch": epoch,
        "task_state_dict": {key: value.detach().cpu() for key, value in task.state_dict().items()},
        "config": config,
        "validation_metrics": metrics,
    }
    if vision_provider is not None and config["vision_mode"] == "lora":
        payload["vision_adapter_state_dict"] = {
            key: value.detach().cpu()
            for key, value in get_peft_model_state_dict(vision_provider.backbone).items()
        }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    if not 1 <= args.subject_id <= 10:
        raise ValueError("subject-id must be in 1..10")
    if args.vision_mode == "frozen" and not args.feature_cache:
        raise ValueError("Frozen mode requires --feature-cache")
    if args.vision_mode == "lora" and args.feature_cache:
        raise ValueError("LoRA mode must use online images, not a frozen feature cache")
    manifest = load_manifest(args.manifest)
    if int(manifest["subject_id"]) != args.subject_id or manifest["split"] != "train":
        raise ValueError("Training manifest subject/split does not match the requested run")
    if args.validation_manifest:
        validation_manifest = load_manifest(args.validation_manifest)
        if (
            int(validation_manifest["subject_id"]) != args.subject_id
            or validation_manifest["split"] != "train"
            or validation_manifest["records_sha256"] != manifest["records_sha256"]
        ):
            raise ValueError("Validation manifest is not the matching training manifest")
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is unavailable")

    vision_provider = None
    processor = None
    if args.vision_mode == "lora":
        vision_provider, processor = load_clip_provider(
            model_path=args.clip_path,
            layer_ids=args.layer_ids,
            vision_mode="lora",
            lora_rank=args.lora_rank,
            device=device,
            dtype=torch.float32,
        )
    train_dataset = ThingsEEGSubjectDataset(
        manifest_path=args.manifest,
        subset=args.subset,
        seed=args.seed,
        feature_cache=args.feature_cache,
        expected_layer_ids=args.layer_ids,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
        collate_fn=SAMGACollator(processor),
    )
    validation_loader = None
    if args.validation_manifest:
        validation_dataset = ThingsEEGSubjectDataset(
            manifest_path=args.validation_manifest,
            subset="pilot_validation",
            seed=args.seed,
            feature_cache=args.validation_feature_cache,
            expected_layer_ids=args.layer_ids,
            smooth_probability=0.0,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            collate_fn=SAMGACollator(processor),
        )

    # Loading/injecting LoRA consumes RNG while frozen-cache setup does not. Reset
    # immediately before task construction so paired Frozen/LoRA cells have
    # bit-identical EEG/projector/router initialization for the same seed.
    seed_everything(args.seed)
    task = SAMGATaskModel(
        layer_ids=args.layer_ids,
        prior_center=args.prior_center,
    ).to(device=device, dtype=torch.float32)
    task_initial_state_sha256 = hash_state_dict(task.state_dict())
    criterion = SAMGALoss(
        eeg_l2norm=args.eeg_l2norm,
        image_l2norm=args.image_l2norm,
    ).to(device)
    optimizer = make_optimizer(
        task,
        vision_provider,
        task_lr=args.stage1_lr,
        vision_lr=args.stage1_lr * args.vision_lr_ratio,
        weight_decay=args.weight_decay,
        include_shared=True,
    )
    candidate_epochs = sorted({int(value) for value in args.candidate_epochs.split(",") if value})
    config = vars(args).copy()
    config.update(
        {
            "schema_version": 1,
            "git_revision": git_revision(PROJECT_ROOT),
            "device_resolved": str(device),
            "torch_version": torch.__version__,
            "train_rows": len(train_dataset),
            "task_trainable_parameters": parameter_count(task.parameters()),
            "task_initial_state_sha256": task_initial_state_sha256,
            "vision_trainable_parameters": (
                parameter_count(vision_provider.parameters()) if vision_provider is not None else 0
            ),
            "manifest_sha256": hash_file(args.manifest),
            "clip_config_sha256": hash_file(Path(args.clip_path) / "config.json"),
        }
    )
    atomic_write_json(output_dir / "run_config.json", config)
    history_path = output_dir / "training_history.jsonl"
    validation_path = output_dir / "validation_metrics.jsonl"
    global_step = 0
    started = time.time()
    first_step_gradient_norms: dict[str, float] | None = None
    for epoch in range(1, args.num_epochs + 1):
        if epoch == args.stage1_epochs + 1:
            task.freeze_shared_encoder()
            optimizer = make_optimizer(
                task,
                vision_provider,
                task_lr=args.stage2_lr,
                vision_lr=args.stage2_lr * args.vision_lr_ratio,
                weight_decay=args.weight_decay,
                include_shared=False,
            )
        task.train()
        if task.shared_frozen:
            task.shared_encoder.eval()
        if vision_provider is not None:
            vision_provider.train()
        epoch_total = 0.0
        epoch_contrast = 0.0
        epoch_mmd = 0.0
        batches = 0
        weight = mmd_weight(epoch, args.stage1_epochs, args.mmd_start, args.mmd_end)
        stop = False
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            eeg = batch["eeg"].to(device=device, dtype=torch.float32, non_blocking=True)
            subject_ids = batch["subject_ids"].to(device=device, non_blocking=True)
            layers = vision_features(batch, vision_provider, device)
            eeg_features, image_features, _ = task(eeg, layers, subject_ids)
            loss = criterion(eeg_features, image_features, mmd_weight=weight)
            if not torch.isfinite(loss.total):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {global_step}")
            loss.total.backward()
            if first_step_gradient_norms is None:
                first_step_gradient_norms = {
                    "task": gradient_norm(task.parameters()),
                    "vision": (
                        gradient_norm(vision_provider.parameters())
                        if vision_provider is not None
                        else 0.0
                    ),
                }
            optimizer.step()
            epoch_total += float(loss.total.detach())
            epoch_contrast += float(loss.contrastive.detach())
            epoch_mmd += float(loss.mmd.detach())
            batches += 1
            global_step += 1
            if args.max_train_steps is not None and global_step >= args.max_train_steps:
                stop = True
                break
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": epoch_total / max(batches, 1),
            "contrastive_loss": epoch_contrast / max(batches, 1),
            "mmd_loss": epoch_mmd / max(batches, 1),
            "mmd_weight": weight,
            "router_global_weights": [float(value) for value in task.router.global_weights().detach().cpu()],
            "elapsed_seconds": time.time() - started,
        }
        if epoch == 1:
            record["first_step_gradient_norms"] = first_step_gradient_norms
        append_jsonl(history_path, record)
        validation_metrics = None
        if validation_loader is not None and (epoch in candidate_epochs or stop):
            validation_metrics = validate(task, vision_provider, validation_loader, device)
            validation_metrics.update({"epoch": epoch, "global_step": global_step})
            append_jsonl(validation_path, validation_metrics)
        if epoch in candidate_epochs or stop or epoch == args.num_epochs:
            save_checkpoint(
                output_dir / f"checkpoint_epoch{epoch:03d}.pt",
                epoch=epoch,
                task=task,
                vision_provider=vision_provider,
                config=config,
                metrics=validation_metrics,
            )
        if stop:
            break
    atomic_write_json(
        output_dir / "completion.json",
        {
            "run_id": args.run_id,
            "completed": True,
            "final_epoch": epoch,
            "global_step": global_step,
            "elapsed_seconds": time.time() - started,
            "first_step_gradient_norms": first_step_gradient_norms,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
    )


if __name__ == "__main__":
    main()
