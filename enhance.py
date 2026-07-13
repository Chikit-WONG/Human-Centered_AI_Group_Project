import os
import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from PIL import ImageOps, PngImagePlugin
from torchvision.transforms.functional import to_pil_image, to_tensor
from accelerate import Accelerator
from accelerate.utils import gather_object
from diffusers import StableDiffusionXLPipeline
from diffusers.training_utils import free_memory

from main.models_clip import BrainModel, BrainCLIPModel
from main.models_diffusion import IPAdapterModel
from main.data import (
    load_embedding_dataset,
    load_things_brain_dataset,
    load_image_dataset,
)
from utils.utils_eval import eval_images

MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


# =========================
# Hard-coded settings
# =========================

MIXED_PRECISION = "bf16"
WEIGHT_DTYPE = torch.bfloat16
BANK_DTYPE = torch.float16

# Image-bank embeddings are loaded directly from embedding datasets.
# No image re-encoding, no safetensors cache, and no merge-by-image-id.
BRAIN_KEY = "eeg"
EMBEDDING_DIRECTORY = "/home/jiawen/data/things-eeg/embeddings"  # change to your embedding root
DATASET_NAME = "things"
IMAGE_SPLIT = "train"
CACHE_DIR = ".cache"

SUBJECT_ID = "8"
BASE_VISION_PATH = "/home/jiawen/pretrained/laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FROZEN_VISION_PATH = "/home/jiawen/pretrained/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
CHECKPOINT_PATH = (
    f"/home/jiawen/exp/clip-things-eeg-subj{SUBJECT_ID}-"
    "laion-CLIP-ViT-B-32-laion2B-s34B-b79K-r32"
)

# The adapted embedding key should match how you built the embedding dataset.
# For the PEFT-adapted CLIP vision encoder, this is usually the adapter subfolder.
ADAPTED_EMBEDDING_KEY = f"laion-CLIP-ViT-B-32-laion2B-s34B-b79K_adapted_{BRAIN_KEY}_subj{SUBJECT_ID}"
FROZEN_EMBEDDING_KEY = FROZEN_VISION_PATH

# Optional extra bank, e.g. ImageNet or another external image pool.
EXTRA_EMBEDDING_DIRECTORY = "/home/jiawen/data/visual-layer/imagenet-1k-vl-enriched/embeddings"
EXTRA_DATASET_NAME = "imagenet-1k"
EXTRA_IMAGE_SPLIT = "train"
MAX_EXTRA_SAMPLES = None

BRAIN_DIRECTORY = "/home/jiawen/data/things-eeg/Preprocessed_data_250Hz_whiten"
THINGS_IMAGE_DIRECTORY = "/home/jiawen/data/things-eeg"
SELECTED_CHANNELS = "P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,PO3,POz,PO4,PO8,O1,Oz,O2"

IP_ADAPTER_PATH = (
    "/home/jiawen/pretrained/ip-adapter/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"
)
SDXL_PATH = "/home/jiawen/pretrained/stabilityai/sdxl-turbo"

EMBED_BATCH_SIZE = 8192
BRAIN_BATCH_SIZE = 100
GEN_BATCH_SIZE = 8
NUM_GENERATIONS_PER_SAMPLE = 10
NUM_WORKERS = 8

QUERY_INDEX = 109

# Use a larger top-M neighborhood for training-free positive/negative decomposition.
# PRINT_TOPK only controls logging.
TOPK = 128
PRINT_TOPK = 8
RETRIEVAL_CHUNK_SIZE = 65536

# =========================
# Contrastive Neighborhood Refinement v2 (training-free)
# =========================

# CNR-v2 uses adapted-space relevance (QIR proxy) and frozen-space consistency (GRC proxy)
# to decompose the retrieved neighborhood into positive and negative visual memories.
# It then applies CFG-like embedding arithmetic before IP-Adapter generation:
#   z_cond = normalize(z_pos + lambda * (z_pos - z_neg)).
USE_CNR_V2 = True
CNR_POSITIVE_QIR_SCALE = 0.5
CNR_POSITIVE_GRC_SCALE = 1.0
CNR_NEGATIVE_GUIDANCE_SCALE = 0.6
CNR_NEGATIVE_GRC_POWER = 1.0
CNR_EPS = 1e-6

# Keep the final condition near the proven vanilla condition.  1.0 means use the
# full CNR-v2 refined direction.  Lower values are useful if the negative branch
# becomes too aggressive on a new subject/checkpoint.
CNR_REFINEMENT_STRENGTH = 1.0

WIDTH = 512
HEIGHT = 512
NUM_INFERENCE_STEPS = 4
PROMPT = "best quality, high quality"

MAX_EVAL_SAMPLES = None  # None means all brain test samples.
EVAL_CHUNK_SIZE = 200
SAVE_GENERATED_IMAGES = True
SAVE_DIR = "test_outputs"

# Full list comparison is still much faster than merge-by-id, but for very large
# banks a sampled sanity check is usually enough because embedding builders do
# not shuffle within the same dataset/split.
FULL_ORDER_CHECK = False


# =========================
# Helpers
# =========================


def pil_to_eval_tensor(image):
    image = ImageOps.fit(image.convert("RGB"), (WIDTH, HEIGHT))
    return to_tensor(image)


def load_mix_temperature() -> float:
    """Load the CLIP/brain alignment temperature from the trained BrainCLIPModel."""
    pretrained_model = BrainCLIPModel.from_pretrained(
        CHECKPOINT_PATH, subfolder="brain_model"
    )
    mix_temperature = float(pretrained_model.logit_scale.detach().item())
    del pretrained_model
    return mix_temperature


def bank_specs() -> Iterable[Tuple[str, str, str]]:
    yield EMBEDDING_DIRECTORY, DATASET_NAME, IMAGE_SPLIT
    if EXTRA_DATASET_NAME is not None:
        yield (
            EXTRA_EMBEDDING_DIRECTORY or EMBEDDING_DIRECTORY,
            EXTRA_DATASET_NAME,
            EXTRA_IMAGE_SPLIT,
        )


def sampled_positions(length: int) -> List[int]:
    if length <= 0:
        return []
    positions = {0, length - 1, length // 2, length // 3, (2 * length) // 3}
    return sorted(pos for pos in positions if 0 <= pos < length)


def assert_same_order(
    adapted_ids: List[str],
    frozen_ids: List[str],
    dataset_name: str,
    split: str,
) -> None:
    if len(adapted_ids) != len(frozen_ids):
        raise ValueError(
            f"Embedding length mismatch for {dataset_name}/{split}: "
            f"adapted={len(adapted_ids)} vs frozen={len(frozen_ids)}."
        )

    if FULL_ORDER_CHECK:
        if adapted_ids != frozen_ids:
            for idx, (a, b) in enumerate(zip(adapted_ids, frozen_ids)):
                if a != b:
                    raise ValueError(
                        f"Embedding order mismatch for {dataset_name}/{split} at index {idx}: "
                        f"adapted={a}, frozen={b}."
                    )
    else:
        for idx in sampled_positions(len(adapted_ids)):
            if adapted_ids[idx] != frozen_ids[idx]:
                raise ValueError(
                    f"Embedding order mismatch for {dataset_name}/{split} at sampled index {idx}: "
                    f"adapted={adapted_ids[idx]}, frozen={frozen_ids[idx]}. "
                    "Set FULL_ORDER_CHECK=True for exhaustive debugging."
                )


def load_embedding_table(
    embedding_directory: str,
    dataset_name: str,
    split: str,
    embedding_key: str,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, List[str]]:
    ds = load_embedding_dataset(
        embedding_directory=embedding_directory,
        dataset_name=dataset_name,
        split=split,
        model_key=embedding_key,
        cache_dir=CACHE_DIR,
    )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    def collate(examples):
        image_ids = [str(ex["image_id"]) for ex in examples]
        embs = torch.stack(
            [
                torch.as_tensor(ex["emb"], dtype=torch.float32).flatten()
                for ex in examples
            ],
            dim=0,
        )
        return {"image_ids": image_ids, "embs": embs}

    loader = DataLoader(
        ds,
        batch_size=EMBED_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate,
    )

    all_embs: List[torch.Tensor] = []
    all_ids: List[str] = []
    desc = f"Load embeddings: {dataset_name}/{split} [{embedding_key}]"
    for batch in tqdm(loader, desc=desc):
        all_embs.append(batch["embs"].to(BANK_DTYPE).cpu())
        all_ids.extend(batch["image_ids"])

    if not all_embs:
        raise ValueError(
            f"Empty embedding dataset: {dataset_name}/{split}, key={embedding_key}"
        )
    return torch.cat(all_embs, dim=0), all_ids


@torch.no_grad()
def load_image_bank_from_embedding_datasets() -> (
    Tuple[torch.Tensor, torch.Tensor, List[str]]
):
    adapted_parts: List[torch.Tensor] = []
    frozen_parts: List[torch.Tensor] = []
    id_parts: List[str] = []

    for embedding_directory, dataset_name, split in bank_specs():
        max_samples = MAX_EXTRA_SAMPLES if dataset_name == EXTRA_DATASET_NAME else None
        adapted_embeds, adapted_ids = load_embedding_table(
            embedding_directory=embedding_directory,
            dataset_name=dataset_name,
            split=split,
            embedding_key=ADAPTED_EMBEDDING_KEY,
            max_samples=max_samples,
        )
        frozen_embeds, frozen_ids = load_embedding_table(
            embedding_directory=embedding_directory,
            dataset_name=dataset_name,
            split=split,
            embedding_key=FROZEN_EMBEDDING_KEY,
            max_samples=max_samples,
        )

        assert_same_order(
            adapted_ids, frozen_ids, dataset_name=dataset_name, split=split
        )
        adapted_parts.append(adapted_embeds)
        frozen_parts.append(frozen_embeds)
        id_parts.extend(adapted_ids)

    image_embeds = torch.cat(adapted_parts, dim=0).float()
    frozen_embeds = torch.cat(frozen_parts, dim=0).float()

    if image_embeds.size(0) != frozen_embeds.size(0) or image_embeds.size(0) != len(
        id_parts
    ):
        raise ValueError(
            f"Bank size mismatch: adapted={image_embeds.size(0)}, "
            f"frozen={frozen_embeds.size(0)}, ids={len(id_parts)}."
        )

    print(
        f"Loaded image bank: size={len(id_parts)}, "
        f"adapted_dim={image_embeds.shape[-1]}, frozen_dim={frozen_embeds.shape[-1]}"
    )
    return image_embeds, frozen_embeds, id_parts


# =========================
# Brain embeds
# =========================


@torch.no_grad()
def encode_brain_testset(accelerator: Accelerator):
    brain_model = BrainModel.from_pretrained(CHECKPOINT_PATH, subfolder="brain_model")
    brain_model.to(accelerator.device, WEIGHT_DTYPE).eval()

    brain_dataset = load_things_brain_dataset(
        data_directory=BRAIN_DIRECTORY,
        subject_ids=SUBJECT_ID,
        selected_channels=SELECTED_CHANNELS,
        brain_column=BRAIN_KEY,
        split="test",
    )
    brain_dataset.set_format("torch")

    def collate_brain(examples):
        subject_ids = torch.tensor(
            [ex["subject_id"] for ex in examples], dtype=torch.long
        ).contiguous()
        image_ids = [str(ex["image_id"]) for ex in examples]
        brain = torch.stack([ex[BRAIN_KEY] for ex in examples]).float().contiguous()
        return {
            "brain_signals": brain,
            "subject_ids": subject_ids,
            "image_ids": image_ids,
        }

    dataloader = DataLoader(
        brain_dataset,
        batch_size=BRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_brain,
    )
    dataloader = accelerator.prepare(dataloader)

    all_brain_embeds = []
    all_image_ids = []

    for batch in tqdm(
        dataloader,
        disable=not accelerator.is_local_main_process,
        desc="Encode brain testset",
    ):
        brain_signals = batch["brain_signals"].to(accelerator.device, WEIGHT_DTYPE)
        subject_ids = batch["subject_ids"].to(accelerator.device)

        brain_embeds = brain_model(brain_signals, subject_ids=subject_ids)
        brain_embeds = accelerator.gather_for_metrics(brain_embeds)
        image_ids = gather_object(batch["image_ids"])

        # gather_for_metrics / gather_object return the gathered mini-batch on every process.
        # Keeping the gathered results on every rank makes later generation sharding simple.
        all_brain_embeds.append(brain_embeds.detach().cpu().float())
        all_image_ids.extend(image_ids)

    accelerator.wait_for_everyone()

    del brain_model
    free_memory()

    return torch.cat(all_brain_embeds, dim=0), all_image_ids


# =========================
# Retrieval + training-free refiner
# =========================


def minmax_normalize(x: torch.Tensor, dim: int = -1, eps: float = CNR_EPS) -> torch.Tensor:
    x_min = x.min(dim=dim, keepdim=True).values
    x_max = x.max(dim=dim, keepdim=True).values
    return (x - x_min) / (x_max - x_min).clamp_min(eps)


def normalize_weights(weights: torch.Tensor, dim: int = -1, eps: float = CNR_EPS) -> torch.Tensor:
    weights = weights.clamp_min(0.0)
    denom = weights.sum(dim=dim, keepdim=True)
    fallback = torch.full_like(weights, 1.0 / weights.size(dim))
    return torch.where(denom > eps, weights / denom.clamp_min(eps), fallback)


def normalized_entropy(probs: torch.Tensor, dim: int = -1, eps: float = CNR_EPS) -> torch.Tensor:
    if probs.size(dim) <= 1:
        return probs.sum(dim=dim) * 0.0
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=dim)
    return entropy / math.log(probs.size(dim))


@torch.no_grad()
def contrastive_neighborhood_refinement_v2(
    topk_scores: torch.Tensor,
    retrieved_frozen: torch.Tensor,
    logit_scale: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build a training-free IP-Adapter condition from retrieved frozen embeddings.

    topk_scores are cosine similarities in the adapted EEG-image space and act as
    the QIR proxy. retrieved_frozen lives in the frozen image/IP-Adapter space and
    is used for GRC-style visual consistency and final generation.

    The refiner extracts:
      - z_pos: EEG-supported and visually coherent memory,
      - z_neg: visually coherent but EEG-unsupported memory,
    and returns a CFG-like condition z_pos + lambda * (z_pos - z_neg).
    """
    scores = topk_scores.float()
    frozen = retrieved_frozen.float()
    batch_size, num_neighbors, _ = frozen.shape

    # Vanilla score-softmax mixture. This is also the fallback when CNR-v2 is disabled.
    logit_scale = math.exp(logit_scale)
    rank_probs = (scores * logit_scale).softmax(dim=-1)
    vanilla = torch.sum(rank_probs.unsqueeze(-1) * frozen, dim=1)

    if not USE_CNR_V2 or num_neighbors <= 1:
        zeros = torch.zeros(batch_size, dtype=frozen.dtype, device=frozen.device)
        stats = {
            "rank_probs": rank_probs,
            "pos_weights": rank_probs,
            "neg_weights": torch.zeros_like(rank_probs),
            "guidance_scale": zeros,
            "entropy": normalized_entropy(rank_probs),
            "pos_qir": torch.sum(rank_probs * scores, dim=-1),
            "neg_qir": zeros,
            "pos_grc": zeros,
            "neg_grc": zeros,
            "vanilla_frozen_embeds": vanilla,
        }
        return vanilla, stats

    # QIR proxy: adapted-space EEG relevance. We keep the absolute scores for
    # logging, but use min-max values for robust intra-neighborhood decomposition.
    qir = minmax_normalize(scores, dim=-1)

    # GRC proxy: local visual consistency in the frozen generation space. A sample
    # receives high density if it belongs to a coherent visual mode among top-M.
    frozen_unit = F.normalize(frozen, dim=-1)
    sim = torch.einsum("bmd,bnd->bmn", frozen_unit, frozen_unit).clamp(-1.0, 1.0)
    eye = torch.eye(num_neighbors, dtype=torch.bool, device=sim.device).unsqueeze(0)
    positive_sim = sim.clamp_min(0.0).masked_fill(eye, 0.0)
    non_self_mass = (rank_probs.unsqueeze(1) * (~eye).to(rank_probs.dtype)).sum(dim=-1)
    grc = (rank_probs.unsqueeze(1) * positive_sim).sum(dim=-1) / non_self_mass.clamp_min(CNR_EPS)
    grc = minmax_normalize(grc, dim=-1)

    # Positive memory: keep candidates that are both EEG-relevant and visually coherent.
    pos_logits = (
        rank_probs.clamp_min(CNR_EPS).log()
        + CNR_POSITIVE_QIR_SCALE * qir
        + CNR_POSITIVE_GRC_SCALE * grc
    )
    pos_weights = pos_logits.softmax(dim=-1)
    pos_raw = torch.sum(pos_weights.unsqueeze(-1) * frozen, dim=1)
    pos_dir = F.normalize(pos_raw, dim=-1)

    # Negative memory: coherent but EEG-unsupported alternatives inside the same top-M.
    # The separation term prevents subtracting samples already aligned with z_pos.
    sep = (1.0 - torch.einsum("bmd,bd->bm", frozen_unit, pos_dir)).clamp_min(0.0)
    neg_scores = (
        rank_probs
        * (1.0 - qir).clamp_min(0.0)
        * grc.clamp_min(0.0).pow(CNR_NEGATIVE_GRC_POWER)
        * sep
    )

    # If the negative score degenerates, use a soft fallback that still prefers
    # low-QIR, high-GRC, separated candidates, but let lambda decide whether to use it.
    fallback_logits = (1.0 - qir) + grc + sep
    fallback_weights = fallback_logits.softmax(dim=-1)
    neg_mass = neg_scores.sum(dim=-1, keepdim=True)
    neg_weights = torch.where(
        neg_mass > CNR_EPS,
        neg_scores / neg_mass.clamp_min(CNR_EPS),
        fallback_weights,
    )
    neg_raw = torch.sum(neg_weights.unsqueeze(-1) * frozen, dim=1)
    neg_dir = F.normalize(neg_raw, dim=-1)

    entropy = normalized_entropy(rank_probs)
    pos_qir = torch.sum(pos_weights * scores, dim=-1)
    neg_qir = torch.sum(neg_weights * scores, dim=-1)
    pos_grc = torch.sum(pos_weights * grc, dim=-1)
    neg_grc = torch.sum(neg_weights * grc, dim=-1)

    # The guidance is strong only when the neighborhood is multi-modal, the negative
    # mode is coherent, and the positive mode is more EEG-relevant than the negative.
    qir_gap = minmax_normalize(torch.stack([neg_qir, pos_qir], dim=-1), dim=-1)[:, 1]
    guidance = CNR_NEGATIVE_GUIDANCE_SCALE * entropy * neg_grc * qir_gap
    guidance = guidance.clamp(0.0, CNR_NEGATIVE_GUIDANCE_SCALE)

    refined_dir = F.normalize(pos_dir + guidance.unsqueeze(-1) * (pos_dir - neg_dir), dim=-1)
    vanilla_dir = F.normalize(vanilla, dim=-1)
    strength = float(CNR_REFINEMENT_STRENGTH)
    cond_dir = F.normalize((1.0 - strength) * vanilla_dir + strength * refined_dir, dim=-1)
    # Preserve the embedding scale expected by the pretrained IP-Adapter.
    # The scale follows the positive evidence rather than the unit direction.
    cond = cond_dir * pos_raw.norm(dim=-1, keepdim=True).clamp_min(CNR_EPS)

    stats = {
        "rank_probs": rank_probs,
        "pos_weights": pos_weights,
        "neg_weights": neg_weights,
        "guidance_scale": guidance,
        "entropy": entropy,
        "pos_qir": pos_qir,
        "neg_qir": neg_qir,
        "pos_grc": pos_grc,
        "neg_grc": neg_grc,
        "vanilla_frozen_embeds": vanilla,
    }
    return cond, stats


@torch.no_grad()
def retrieve_topk_batch(
    query_embeds: torch.Tensor,
    image_embeds: torch.Tensor,
    frozen_embeds: torch.Tensor,
    image_ids: List[str],
    topk: int,
    temperature: float,
    device: torch.device,
):
    query = F.normalize(query_embeds.float().to(device), dim=-1)

    global_scores = None
    global_indices = None

    for start in tqdm(
        range(0, image_embeds.size(0), RETRIEVAL_CHUNK_SIZE), desc="Retrieve"
    ):
        end = min(start + RETRIEVAL_CHUNK_SIZE, image_embeds.size(0))
        gallery = F.normalize(image_embeds[start:end].float().to(device), dim=-1)

        scores = query @ gallery.T
        local_scores, local_indices = scores.topk(min(topk, end - start), dim=-1)
        local_indices = local_indices + start

        if global_scores is None:
            global_scores, global_indices = local_scores, local_indices
        else:
            merged_scores = torch.cat([global_scores, local_scores], dim=-1)
            merged_indices = torch.cat([global_indices, local_indices], dim=-1)
            global_scores, order = merged_scores.topk(topk, dim=-1)
            global_indices = merged_indices.gather(dim=-1, index=order)

    topk_scores = global_scores.cpu()
    topk_indices = global_indices.cpu()
    retrieved_frozen = frozen_embeds[topk_indices].float()

    merged_frozen, refiner_stats = contrastive_neighborhood_refinement_v2(
        topk_scores=topk_scores,
        retrieved_frozen=retrieved_frozen,
        logit_scale=temperature,
    )
    mix_weights = refiner_stats["rank_probs"]
    topk_image_ids = [[image_ids[i] for i in row.tolist()] for row in topk_indices]

    return {
        "topk_scores": topk_scores,
        "topk_indices": topk_indices,
        "topk_image_ids": topk_image_ids,
        "mix_weights": mix_weights,
        "merged_frozen_embeds": merged_frozen,
        "vanilla_frozen_embeds": refiner_stats["vanilla_frozen_embeds"],
        "refiner_stats": refiner_stats,
    }


# =========================
# GT images
# =========================


def load_things_test_image_map() -> Dict[str, torch.Tensor]:
    test_images = load_image_dataset(
        dataset_name="things",
        image_directory=THINGS_IMAGE_DIRECTORY,
        split="test",
        image_decode=True,
    )

    image_map = {}
    for ex in tqdm(test_images, desc="Load THINGS GT images"):
        image_map[str(ex["image_id"])] = pil_to_eval_tensor(ex["image"])

    return image_map


def shard_bounds(num_items: int, process_index: int, num_processes: int) -> Tuple[int, int]:
    start = (num_items * process_index) // num_processes
    end = (num_items * (process_index + 1)) // num_processes
    return start, end


# =========================
# Generation + evaluation
# =========================


@torch.no_grad()
def generate_images_from_embeddings(
    accelerator: Accelerator,
    merged_frozen_embeds: torch.Tensor,
    ip_adapter: IPAdapterModel,
    pipeline: StableDiffusionXLPipeline,
    group_index: int,
    save_start_index: int = 0,
):
    fake_tensors = []

    if SAVE_GENERATED_IMAGES:
        os.makedirs(SAVE_DIR, exist_ok=True)

    for start in tqdm(
        range(0, merged_frozen_embeds.size(0), GEN_BATCH_SIZE),
        desc=f"Generate group {group_index + 1}/{NUM_GENERATIONS_PER_SAMPLE}",
    ):
        end = min(start + GEN_BATCH_SIZE, merged_frozen_embeds.size(0))
        batch_embeds = merged_frozen_embeds[start:end].to(
            accelerator.device, WEIGHT_DTYPE
        )
        ip_hidden_states = ip_adapter(batch_embeds)

        images = pipeline(
            prompt=[PROMPT] * batch_embeds.size(0),
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=0.0,
            num_inference_steps=NUM_INFERENCE_STEPS,
            cross_attention_kwargs={"ip_hidden_states": ip_hidden_states},
        ).images

        generated = torch.stack(
            [pil_to_eval_tensor(image) for image in images], dim=0
        ).clamp(0.0, 1.0)

        for local_idx, tensor in enumerate(generated):
            if SAVE_GENERATED_IMAGES:
                global_idx = save_start_index + start + local_idx
                to_pil_image(tensor).save(
                    os.path.join(
                        SAVE_DIR, f"fake_group{group_index:02d}_{global_idx:04d}.png"
                    )
                )
            fake_tensors.append(tensor)

    return torch.stack(fake_tensors, dim=0)


def weighted_average_chunk_metrics(
    all_chunk_metrics: List[Tuple[int, Dict[str, float]]],
) -> Dict[str, float]:
    total = sum(chunk_size for chunk_size, _ in all_chunk_metrics)
    merged_metrics = {}
    for key in all_chunk_metrics[0][1].keys():
        merged_metrics[key] = (
            sum(chunk_size * metrics[key] for chunk_size, metrics in all_chunk_metrics)
            / total
        )
    return merged_metrics


def average_group_metrics(
    all_group_metrics: List[Dict[str, float]],
) -> Dict[str, float]:
    merged_metrics = {}
    for key in all_group_metrics[0].keys():
        merged_metrics[key] = sum(metrics[key] for metrics in all_group_metrics) / len(
            all_group_metrics
        )
    return merged_metrics


@torch.no_grad()
def generate_and_eval_in_chunks(
    accelerator: Accelerator,
    merged_frozen_embeds: torch.Tensor,
    image_ids: List[str],
    gt_map: Dict[str, torch.Tensor],
    global_num_samples: int,
    global_start_index: int = 0,
):
    """Generate local shard on each GPU and gather images for global evaluation.

    Each process receives only its own contiguous shard of conditioning embeddings.
    SDXL/IP-Adapter are replicated across GPUs, so generation is data-parallel.
    """
    if merged_frozen_embeds.size(0) != len(image_ids):
        raise ValueError(
            f"Conditioning embeddings and image_ids must be one-to-one matched, "
            f"but got cond={merged_frozen_embeds.size(0)} and image_ids={len(image_ids)}."
        )

    ip_adapter = IPAdapterModel.from_ip_adapter(IP_ADAPTER_PATH)
    pipeline = StableDiffusionXLPipeline.from_pretrained(SDXL_PATH)

    ip_adapter.to(accelerator.device, WEIGHT_DTYPE).eval()
    pipeline.to(accelerator.device, WEIGHT_DTYPE)
    ip_adapter.bind_unet(pipeline.unet)
    # hide the progress bar inside the pipeline
    pipeline.set_progress_bar_config(disable=True)

    local_num_samples = merged_frozen_embeds.size(0)
    accelerator.print(
        f"\nRun distributed generation + evaluation: total={global_num_samples}, "
        f"world_size={accelerator.num_processes}, local_batch={local_num_samples}, "
        f"groups={NUM_GENERATIONS_PER_SAMPLE}."
    )

    all_group_metrics: List[Dict[str, float]] = []

    for group_index in range(NUM_GENERATIONS_PER_SAMPLE):
        local_real_parts: List[torch.Tensor] = []
        local_fake_parts: List[torch.Tensor] = []

        iterator = range(0, local_num_samples, EVAL_CHUNK_SIZE)
        for chunk_start in tqdm(
            iterator,
            disable=not accelerator.is_local_main_process,
            desc=f"Generate group {group_index + 1}/{NUM_GENERATIONS_PER_SAMPLE} [rank {accelerator.process_index}]",
        ):
            chunk_end = min(chunk_start + EVAL_CHUNK_SIZE, local_num_samples)
            chunk_image_ids = image_ids[chunk_start:chunk_end]
            cond_chunk = merged_frozen_embeds[chunk_start:chunk_end]
            real_chunk = torch.stack([gt_map[image_id] for image_id in chunk_image_ids], dim=0)

            fake_chunk = generate_images_from_embeddings(
                accelerator=accelerator,
                merged_frozen_embeds=cond_chunk,
                ip_adapter=ip_adapter,
                pipeline=pipeline,
                group_index=group_index,
                save_start_index=global_start_index + chunk_start,
            )

            if real_chunk.size(0) != fake_chunk.size(0):
                raise ValueError(
                    f"Group {group_index}, local chunk [{chunk_start}, {chunk_end}) is not one-to-one matched: "
                    f"real={real_chunk.size(0)}, fake={fake_chunk.size(0)}."
                )

            local_real_parts.append(real_chunk.cpu())
            local_fake_parts.append(fake_chunk.cpu())

            del real_chunk, fake_chunk, cond_chunk
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()

        if local_real_parts:
            local_real = torch.cat(local_real_parts, dim=0)
            local_fake = torch.cat(local_fake_parts, dim=0)
        else:
            local_real = torch.empty(0, 3, HEIGHT, WIDTH)
            local_fake = torch.empty(0, 3, HEIGHT, WIDTH)

        gathered_real_parts = gather_object([local_real])
        gathered_fake_parts = gather_object([local_fake])

        if accelerator.is_main_process:
            real_images = torch.cat([x for x in gathered_real_parts if x.numel() > 0], dim=0)
            fake_images = torch.cat([x for x in gathered_fake_parts if x.numel() > 0], dim=0)

            if real_images.size(0) != global_num_samples or fake_images.size(0) != global_num_samples:
                raise ValueError(
                    f"Gathered image count mismatch: real={real_images.size(0)}, "
                    f"fake={fake_images.size(0)}, expected={global_num_samples}."
                )

            metrics = eval_images(
                real_images=real_images.to(accelerator.device),
                fake_images=fake_images.to(accelerator.device),
                device=accelerator.device,
            )
            all_group_metrics.append(metrics)

            accelerator.print(f"\nEvaluation metrics for group {group_index + 1}:")
            for key, value in metrics.items():
                accelerator.print(f"  {key}: {value:.6f}")

        accelerator.wait_for_everyone()

    final_metrics = None
    if accelerator.is_main_process:
        final_metrics = average_group_metrics(all_group_metrics)
        accelerator.print(f"\nFinal metrics averaged over {NUM_GENERATIONS_PER_SAMPLE} groups:")
        for key, value in final_metrics.items():
            accelerator.print(f"  {key}: {value:.6f}")

    del ip_adapter, pipeline
    free_memory()
    accelerator.wait_for_everyone()
    return final_metrics


# =========================
# Main
# =========================


def main():
    accelerator = Accelerator(mixed_precision=MIXED_PRECISION)

    # Brain encoding is already distributed and then gathered on every process.
    brain_embeds, brain_image_ids = encode_brain_testset(accelerator)

    # Every rank loads the bank and retrieves only its own evaluation shard.
    # This avoids cross-process broadcasting of large tensors while making generation data-parallel.
    with accelerator.main_process_first():
        image_embeds, frozen_embeds, image_bank_ids = load_image_bank_from_embedding_datasets()
    mix_temperature = load_mix_temperature()
    accelerator.print(
        f"Loaded mix_temperature from BrainCLIPModel.logit_scale: {mix_temperature:.6f}"
    )

    if MAX_EVAL_SAMPLES is not None:
        brain_embeds = brain_embeds[:MAX_EVAL_SAMPLES]
        brain_image_ids = brain_image_ids[:MAX_EVAL_SAMPLES]

    gt_map = load_things_test_image_map()
    valid_image_ids: List[str] = []
    valid_brain_indices: List[int] = []
    for i, image_id in enumerate(brain_image_ids):
        if image_id not in gt_map:
            accelerator.print(f"Skip missing GT image_id: {image_id}")
            continue
        valid_image_ids.append(image_id)
        valid_brain_indices.append(i)

    if not valid_brain_indices:
        raise ValueError("No valid samples to evaluate.")

    query_index = min(QUERY_INDEX, len(brain_embeds) - 1)
    if accelerator.is_main_process:
        accelerator.print(f"Query brain index: {query_index}")
        accelerator.print(f"Query brain image_id: {brain_image_ids[query_index]}")
        query_ret = retrieve_topk_batch(
            query_embeds=brain_embeds[query_index : query_index + 1],
            image_embeds=image_embeds,
            frozen_embeds=frozen_embeds,
            image_ids=image_bank_ids,
            topk=TOPK,
            temperature=mix_temperature,
            device=accelerator.device,
        )
        accelerator.print("\nExample top-k retrieval / CNR-v2 decomposition:")
        stats = query_ret["refiner_stats"]
        max_print = min(PRINT_TOPK, len(query_ret["topk_image_ids"][0]))
        for rank in range(max_print):
            iid = query_ret["topk_image_ids"][0][rank]
            score = query_ret["topk_scores"][0, rank].item()
            mix_weight = query_ret["mix_weights"][0, rank].item()
            pos_weight = stats["pos_weights"][0, rank].item()
            neg_weight = stats["neg_weights"][0, rank].item()
            accelerator.print(
                f"  top-{rank + 1}: image_id={iid}, score={score:.6f}, "
                f"rank_w={mix_weight:.6f}, pos_w={pos_weight:.6f}, neg_w={neg_weight:.6f}"
            )
        accelerator.print(
            "  CNR-v2 stats: "
            f"lambda={stats['guidance_scale'][0].item():.6f}, "
            f"entropy={stats['entropy'][0].item():.6f}, "
            f"pos_qir={stats['pos_qir'][0].item():.6f}, "
            f"neg_qir={stats['neg_qir'][0].item():.6f}, "
            f"pos_grc={stats['pos_grc'][0].item():.6f}, "
            f"neg_grc={stats['neg_grc'][0].item():.6f}"
        )

    accelerator.wait_for_everyone()

    global_num_samples = len(valid_brain_indices)
    local_start, local_end = shard_bounds(
        global_num_samples, accelerator.process_index, accelerator.num_processes
    )
    local_brain_indices = valid_brain_indices[local_start:local_end]
    local_image_ids = valid_image_ids[local_start:local_end]
    local_brain_embeds = brain_embeds[local_brain_indices]

    accelerator.print(
        f"\nEval samples: total={global_num_samples}, "
        f"rank={accelerator.process_index}, local=[{local_start}, {local_end}), "
        f"local_num={len(local_image_ids)}"
    )

    local_ret = retrieve_topk_batch(
        query_embeds=local_brain_embeds,
        image_embeds=image_embeds,
        frozen_embeds=frozen_embeds,
        image_ids=image_bank_ids,
        topk=TOPK,
        temperature=mix_temperature,
        device=accelerator.device,
    )

    generate_and_eval_in_chunks(
        accelerator=accelerator,
        merged_frozen_embeds=local_ret["merged_frozen_embeds"],
        image_ids=local_image_ids,
        gt_map=gt_map,
        global_num_samples=global_num_samples,
        global_start_index=local_start,
    )


if __name__ == "__main__":
    main()
