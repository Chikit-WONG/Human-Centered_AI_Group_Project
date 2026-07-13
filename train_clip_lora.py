import argparse
import json
import logging
import math
import os
from tqdm.auto import tqdm
from argparse import Namespace
from typing import Optional, Tuple, Union, List

import datasets
import torch
import transformers
from torch import nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, ProjectConfiguration
from torch.utils.data import DataLoader
from transformers import (
    SchedulerType,
    get_scheduler,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from transformers.models.clip.modeling_clip import contrastive_loss
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate.utils import is_wandb_available, is_swanlab_available

from main.data import (
    load_image_dataset,
    load_things_brain_dataset,
    merge_datasets_by_image_id,
    parse_selected_channels,
)

from main.models_clip import BrainCLIPModel, BrainCLIPConfig, BrainCLIPOutput
from utils.utils_training import rotate_checkpoints

wandb = None
if is_wandb_available():
    try:
        import wandb
    except Exception:
        # Optional trackers must not prevent offline training from starting.
        wandb = None

swanlab = None
if is_swanlab_available():
    try:
        import swanlab
    except Exception:
        # The local SwanLab/protobuf installation is currently incompatible.
        # Keep the dependency optional; this reproduction writes metrics locally.
        swanlab = None

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument("--image_directory", type=str, default=None)
    parser.add_argument("--brain_directory", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=".cache")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument(
        "--brain_column", type=str, default="eeg", choices=["eeg", "meg"]
    )
    parser.add_argument(
        "--avg_trials", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--subject_ids", type=str, default="1")
    parser.add_argument("--eval_subject_ids", type=str, default="1")
    parser.add_argument("--selected_channels", type=str, default=None)
    parser.add_argument("--num_brain_channels", type=int, default=None)
    parser.add_argument("--brain_sequence_length", type=int, default=None)
    parser.add_argument(
        "--time_slice",
        type=str,
        default=None,
        help="Brain temporal window [start, end] used for slicing (in samples) before feeding the backbone.",
    )
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--brain_backbone", type=str, default="brain_mlp")
    parser.add_argument("--extra_dim", type=int, default=1440)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument(
        "--lora_layers", type=str, default=None, help="None means fixed training"
    )

    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="epoch",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--validation_steps",
        type=str,
        default="epoch",
        help="Whether to perform validation at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="If passed, will set gradient checkpointing to `True` to save memory.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=256,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=100,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--vision_learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate of the vision part (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        help="Whether or not to use mixed precision training. Choose from 'no','fp16','bf16' or 'fp8'. Will default to"
        "the value in the environment variable `ACCELERATE_MIXED_PRECISION`, which will use the default value in the"
        "accelerate config of the current system or the flag passed with the `accelerate.launch` command. 'fp8'"
        "requires the installation of transformers-engine.",
        choices=["no", "fp16", "bf16", "fp8"],
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="The maximum number of total saved states to keep.",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--metrics_jsonl",
        type=str,
        default=None,
        help="Optional local JSONL file for one validation record per evaluation.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
        ),
    )
    args = parser.parse_args()

    return args


class EnsembleModel(nn.Module):
    def __init__(
        self,
        brain_model: BrainCLIPModel,
        vision_model: Union[PeftModel, CLIPVisionModelWithProjection],
    ):
        super().__init__()

        self.brain_model = brain_model
        self.vision_model = vision_model

        self.train_vision_model = isinstance(vision_model, PeftModel)

    def forward(
        self,
        brain_signals: torch.FloatTensor,
        pixel_values: torch.FloatTensor,
        subject_ids: Optional[torch.LongTensor] = None,
    ) -> BrainCLIPOutput:
        if self.train_vision_model:
            image_embeds = self.vision_model(pixel_values)[0]
        else:
            with torch.no_grad():
                image_embeds = self.vision_model(pixel_values)[0]

        return self.brain_model(
            brain_signals,
            image_embeds=image_embeds,
            subject_ids=subject_ids,
            return_loss=True,
        )


def setup_model(
    args: Namespace,
) -> Tuple[BrainCLIPModel, CLIPVisionModelWithProjection, CLIPImageProcessor]:
    vision_model = CLIPVisionModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path
    )
    if args.gradient_checkpointing:
        vision_model.gradient_checkpointing_enable()

    image_transforms = CLIPImageProcessor.from_pretrained(
        args.pretrained_model_name_or_path
    )

    time_slice = (
        [int(ti.strip()) for ti in args.time_slice.split(",")]
        if args.time_slice is not None
        else None
    )
    selected_channels = parse_selected_channels(args.selected_channels)

    config = BrainCLIPConfig(
        brain_backbone=args.brain_backbone,
        num_brain_channels=(
            len(selected_channels)
            if selected_channels is not None
            else args.num_brain_channels
        ),
        brain_sequence_length=(
            time_slice[1] - time_slice[0]
            if time_slice is not None
            else args.brain_sequence_length
        ),
        embed_dim=vision_model.config.projection_dim,
        extra_dim=args.extra_dim,
        dropout=args.dropout,
    )

    brain_model = BrainCLIPModel(config)

    return brain_model, vision_model, image_transforms


def build_datasets(
    args: Namespace, is_main_process: bool = False
) -> Tuple[datasets.Dataset, Optional[datasets.Dataset]]:
    train_dataset = load_things_brain_dataset(
        data_directory=args.brain_directory,
        split="train",
        subject_ids=args.subject_ids,
        brain_column=args.brain_column,
        avg_trials=args.avg_trials,
        selected_channels=args.selected_channels,
    )

    image_ds = load_image_dataset(
        dataset_name=args.dataset_name,
        image_directory=args.image_directory,
        split="train",
        cache_dir=args.cache_dir,
        image_column=args.image_column,
    ).select_columns([args.image_column, "image_id"])

    train_dataset = merge_datasets_by_image_id(
        train_dataset, image_ds, is_main_process=is_main_process
    )
    train_dataset = train_dataset.cast_column(
        args.image_column, datasets.Image(decode=True)
    )
    train_dataset.set_format(
        "torch", columns=args.brain_column, output_all_columns=True
    )

    if args.eval_ratio <= 0:
        return train_dataset, None

    eval_dataset = load_things_brain_dataset(
        data_directory=args.brain_directory,
        split="test",
        subject_ids=args.subject_ids,
        brain_column=args.brain_column,
        avg_trials=True,
        selected_channels=args.selected_channels,
    )

    image_ds = load_image_dataset(
        dataset_name=args.dataset_name,
        image_directory=args.image_directory,
        split="test",
        cache_dir=args.cache_dir,
        image_column=args.image_column,
    ).select_columns([args.image_column, "image_id"])

    eval_dataset = merge_datasets_by_image_id(
        eval_dataset, image_ds, is_main_process=is_main_process
    )
    eval_dataset = eval_dataset.cast_column(
        args.image_column, datasets.Image(decode=True)
    )
    eval_dataset.set_format("torch", columns=args.brain_column, output_all_columns=True)

    # split_dataset = train_dataset.train_test_split(
    #     test_size=args.eval_ratio, seed=args.seed, shuffle=True
    # )
    # train_dataset, eval_dataset = split_dataset["train"], split_dataset["test"]

    return train_dataset, eval_dataset


def log_validation(
    eval_dataloader: DataLoader,
    model: EnsembleModel,
    accelerator: Accelerator,
    step: int,
    metrics_jsonl: Optional[str] = None,
):
    logger.info(f"Logging validation at global step {step} ...")
    model.eval()

    all_image_embeds: List[torch.Tensor] = []
    all_brain_embeds: List[torch.Tensor] = []
    all_image_ids: List[str] = []

    for batch in eval_dataloader:
        image_ids: List[str] = batch.pop("image_ids")
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        with torch.inference_mode(), accelerator.autocast():
            model_output = model(**batch)

        image_embeds = model_output.image_embeds.detach().float()
        brain_embeds = model_output.brain_embeds.detach().float()

        # Gather across all processes. `gather_for_metrics` handles the duplicated
        # tail samples introduced by distributed evaluation.
        gathered_image_embeds = accelerator.gather_for_metrics(image_embeds)
        gathered_brain_embeds = accelerator.gather_for_metrics(brain_embeds)
        gathered_image_ids = accelerator.gather_for_metrics(
            image_ids, use_gather_object=True
        )

        all_image_embeds.append(gathered_image_embeds.cpu())
        all_brain_embeds.append(gathered_brain_embeds.cpu())
        all_image_ids.extend(gathered_image_ids)

    image_embeds = torch.cat(all_image_embeds, dim=0).to(accelerator.device)
    brain_embeds = torch.cat(all_brain_embeds, dim=0).to(accelerator.device)

    logit_scale = model.brain_model.logit_scale.exp()

    # Global retrieval matrix: rows are brain queries, columns are image gallery items.
    logits_per_brain = logit_scale * brain_embeds @ image_embeds.t()

    num_samples = logits_per_brain.size(0)
    targets = torch.arange(num_samples, device=logits_per_brain.device)

    loss = contrastive_loss(logits_per_brain)

    k = min(5, num_samples)
    topk_scores, topk_indices = torch.topk(logits_per_brain, k=k, dim=1)

    top1_acc = (topk_indices[:, 0] == targets).float().mean().item()
    top5_acc = (topk_indices == targets[:, None]).any(dim=1).float().mean().item()

    metrics = {
        "step": int(step),
        "num_samples": int(num_samples),
        "loss": float(loss.item()),
        "top1_count": int((topk_indices[:, 0] == targets).sum().item()),
        "top5_count": int(
            (topk_indices == targets[:, None]).any(dim=1).sum().item()
        ),
        "top1_acc": float(top1_acc),
        "top5_acc": float(top5_acc),
    }
    accelerator.print("VALIDATION_METRICS " + json.dumps(metrics, sort_keys=True))

    if accelerator.is_main_process:
        if metrics_jsonl:
            metrics_parent = os.path.dirname(os.path.abspath(metrics_jsonl))
            os.makedirs(metrics_parent, exist_ok=True)
            with open(metrics_jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")

        rows = []
        topk_indices_cpu = topk_indices.detach().cpu().tolist()
        topk_scores_cpu = topk_scores.detach().cpu().tolist()

        for i in range(num_samples):
            pred_idx = topk_indices_cpu[i]
            pred_scores = topk_scores_cpu[i]
            rows.append(
                {
                    "sample_idx": i,
                    "gt_image_id": all_image_ids[i],
                    "top1_image_id": all_image_ids[pred_idx[0]],
                    "top5_image_ids": [all_image_ids[j] for j in pred_idx],
                    "top5_scores": pred_scores,
                }
            )

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_scalar("val/loss", loss.item(), step)
                tracker.writer.add_scalar("val/top1_acc", top1_acc, step)
                tracker.writer.add_scalar("val/top5_acc", top5_acc, step)
                tracker.writer.add_text(
                    "val/top5_image_ids",
                    json.dumps(rows, ensure_ascii=False, indent=2),
                    step,
                )

            elif tracker.name == "wandb":
                if wandb is None:
                    logger.warning("wandb tracker requested but wandb failed to import")
                    continue
                table = wandb.Table(
                    columns=[
                        "sample_idx",
                        "gt_image_id",
                        "top1_image_id",
                        "top5_image_ids",
                        "top5_scores",
                    ],
                    data=[
                        [
                            r["sample_idx"],
                            r["gt_image_id"],
                            r["top1_image_id"],
                            r["top5_image_ids"],
                            r["top5_scores"],
                        ]
                        for r in rows
                    ],
                )
                tracker.log(
                    {
                        "val/loss": loss.item(),
                        "val/top1_acc": top1_acc,
                        "val/top5_acc": top5_acc,
                        "val/top5_image_ids": table,
                    },
                    step=step,
                )

            elif tracker.name == "swanlab":
                if swanlab is None:
                    logger.warning(
                        "swanlab tracker requested but swanlab failed to import"
                    )
                    continue
                tracker.log(
                    {
                        "val/loss": loss.item(),
                        "val/top1_acc": top1_acc,
                        "val/top5_acc": top5_acc,
                        "val/top5_image_ids": swanlab.Text(
                            json.dumps(rows, ensure_ascii=False, indent=2)
                        ),
                    },
                    step=step,
                )

            else:
                logger.warning(f"validation logging not implemented for {tracker.name}")

    model.train()


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.run_name,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        log_with=args.report_to,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    brain_model, vision_model, image_transforms = setup_model(args)

    # cast down and move to the CPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    brain_model.to(accelerator.device, weight_dtype)
    vision_model.to(accelerator.device, weight_dtype)

    if args.lora_layers is not None and args.lora_rank > 0:
        if args.lora_layers != "all-linear":
            target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
        else:  # all linear
            target_modules = set()
            for name, module in vision_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    target_modules.add(name)
            target_modules = list(target_modules)

        lora_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            lora_bias=False,
        )

        vision_model = get_peft_model(vision_model, lora_cfg)

    clip_model = EnsembleModel(brain_model, vision_model)
    logger.info("Model loaded successfully.")

    with accelerator.main_process_first():
        train_dataset, eval_dataset = build_datasets(args, accelerator.is_main_process)

    def collate_fn(examples):
        # pixel values
        images = [ex.pop("image") for ex in examples]
        pixel_values = (
            image_transforms(images, return_tensors="pt")
            .pixel_values.float()
            .contiguous()
        )

        image_ids = [ex["image_id"] for ex in examples]

        subject_ids = torch.tensor(
            [ex.pop("subject_id") for ex in examples], dtype=torch.long
        ).contiguous()

        brain = (
            torch.stack([ex.pop(args.brain_column) for ex in examples])
            .float()
            .contiguous()
        )

        time_slice = (
            [int(ti.strip()) for ti in args.time_slice.split(",")]
            if args.time_slice is not None
            else None
        )
        
        if time_slice is not None:
            brain = brain[:, :, time_slice[0]: time_slice[1]]

        return {
            "pixel_values": pixel_values,
            "brain_signals": brain,
            "image_ids": image_ids,
            "subject_ids": subject_ids,
        }

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    eval_dataloader = None
    if eval_dataset is not None:
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=args.per_device_eval_batch_size,
            num_workers=args.dataloader_num_workers,
        )

    # Optimizer
    vision_params = [p for p in clip_model.vision_model.parameters() if p.requires_grad]
    brain_params = [p for p in clip_model.brain_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": brain_params,
                "lr": args.learning_rate,
            },
            {
                "params": vision_params,
                "lr": args.vision_learning_rate,
            },
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes
        ),
    )

    # Prepare everything with our `accelerator`.
    clip_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            clip_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    validation_steps = args.validation_steps
    if validation_steps is not None and validation_steps.isdigit():
        validation_steps = int(validation_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.report_to is not None:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        run = os.path.split(__file__)[-1].split(".")[0]
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value
        accelerator.init_trackers(
            run,
            experiment_config,
            init_kwargs=(
                {
                    "wandb": {"name": args.run_name},
                    "swanlab": {"experiment_name": args.run_name},
                }
                if args.run_name is not None
                else {}
            ),
        )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, args.num_train_epochs):
        clip_model.train()
        if args.report_to is not None:
            total_loss = 0.0

        if (
            args.resume_from_checkpoint is not None
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        for batch in active_dataloader:
            batch.pop("image_ids")
            with accelerator.accumulate(clip_model):
                with accelerator.autocast():
                    batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                    loss = clip_model(**batch)[0]
                # We keep track of the loss at each epoch
                if args.report_to is not None:
                    total_loss += loss.detach().float()

                if accelerator.sync_gradients:
                    params_to_clip = [
                        p for p in clip_model.parameters() if p.requires_grad
                    ]
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                accelerator.log({"step_loss": loss.detach().item()}, completed_steps)

            if isinstance(checkpointing_steps, int):
                if (
                    completed_steps % checkpointing_steps == 0
                    and accelerator.sync_gradients
                ):
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    rotate_checkpoints(args.output_dir, args.save_total_limit)

            if isinstance(validation_steps, int):
                if (
                    completed_steps % validation_steps == 0
                    and accelerator.sync_gradients
                ):
                    log_validation(
                        eval_dataloader,
                        accelerator.unwrap_model(clip_model),
                        accelerator,
                        completed_steps,
                        args.metrics_jsonl,
                    )
            if completed_steps >= args.max_train_steps:
                break

        if args.report_to is not None:
            accelerator.log(
                {
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            rotate_checkpoints(args.output_dir, args.save_total_limit)

        if args.validation_steps == "epoch":
            log_validation(
                eval_dataloader,
                accelerator.unwrap_model(clip_model),
                accelerator,
                completed_steps,
                args.metrics_jsonl,
            )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model: EnsembleModel = accelerator.unwrap_model(clip_model)

        unwrapped_model.brain_model.save_pretrained(
            os.path.join(args.output_dir, "brain_model")
        )

        if unwrapped_model.train_vision_model:
            unwrapped_model.vision_model.save_pretrained(
                os.path.join(args.output_dir, "vision_model")
            )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
