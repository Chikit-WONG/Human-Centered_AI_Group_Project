from __future__ import annotations

import glob
from collections.abc import Mapping
from pathlib import Path
from tqdm import tqdm
from typing import (
    Any,
    Iterable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
    Optional,
    Tuple,
)

import datasets
import transformers
import numpy as np
import torch
from transformers.utils.logging import get_logger

logger = get_logger(__name__)


THINGS_EEG_CHANNEL_NAME_TO_INDEX: Dict[str, int] = {
    "Fp1": 0,
    "Fp2": 1,
    "AF7": 2,
    "AF3": 3,
    "AFz": 4,
    "AF4": 5,
    "AF8": 6,
    "F7": 7,
    "F5": 8,
    "F3": 9,
    "F1": 10,
    "F2": 11,
    "F4": 12,
    "F6": 13,
    "F8": 14,
    "FT9": 15,
    "FT7": 16,
    "FC5": 17,
    "FC3": 18,
    "FC1": 19,
    "FCz": 20,
    "FC2": 21,
    "FC4": 22,
    "FC6": 23,
    "FT8": 24,
    "FT10": 25,
    "T7": 26,
    "C5": 27,
    "C3": 28,
    "C1": 29,
    "Cz": 30,
    "C2": 31,
    "C4": 32,
    "C6": 33,
    "T8": 34,
    "TP9": 35,
    "TP7": 36,
    "CP5": 37,
    "CP3": 38,
    "CP1": 39,
    "CPz": 40,
    "CP2": 41,
    "CP4": 42,
    "CP6": 43,
    "TP8": 44,
    "TP10": 45,
    "P7": 46,
    "P5": 47,
    "P3": 48,
    "P1": 49,
    "Pz": 50,
    "P2": 51,
    "P4": 52,
    "P6": 53,
    "P8": 54,
    "PO7": 55,
    "PO3": 56,
    "POz": 57,
    "PO4": 58,
    "PO8": 59,
    "O1": 60,
    "Oz": 61,
    "O2": 62,
}
IGNORE_TOKEN_ID = -100
MODEL2DIM = {
    "laion-CLIP-ViT-B-32-laion2B-s34B-b79K": 512,
    "paulgavrikov-synclr_vit_b_16": 768,
    "laion-CLIP-ViT-H-14-laion2B-s32B-b79K": 1024,
    "dreamsim-synclr_vitb16": 768,
    "dreamsim-ensemble": 768,
    "open-clip-RN50": 1024,
}


def resolve_model_key(
    model_key_or_path: str, return_dim: bool = False
) -> Tuple[str, Optional[int]]:
    """Resolve model key.

    Rules:
    - "clip_vit_h14" -> "clip_vit_h14"
    - "/path/to/laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
      -> "laion_CLIP-ViT-B-32-laion2B-s34B-b79K"
    """
    p = Path(model_key_or_path)
    name = p.stem

    dim = None

    if p.parent == Path("."):
        return name, dim

    parent_name = p.parent.name

    resolved = f"{parent_name}-{name}" if parent_name else name

    dim = MODEL2DIM[resolved] if return_dim else None

    return resolved, dim


def ensure_image_id(
    ds: datasets.Dataset, image_decode: bool = False, image_column: str = "image"
) -> datasets.Dataset:
    if "image_id" in ds.column_names:
        return ds
    if image_column not in ds.column_names:
        raise ValueError("Dataset has no 'image' column.")

    image_feature = ds.features[image_column]
    raw_ds = ds.cast_column(image_column, datasets.Image(decode=False))

    def _fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        path = ex[image_column].get("path")
        if not path:
            raise ValueError(
                "Cannot derive image_id because dataset image has no stored path."
            )
        return {"image_id": Path(path).stem}

    raw_ds = raw_ds.map(_fn, desc="ensure_image_id")

    if isinstance(image_feature, datasets.Image) and image_decode:
        raw_ds = raw_ds.cast_column(
            image_column, datasets.Image(mode=image_feature.mode, decode=True)
        )

    return raw_ds


def load_image_dataset(
    *,
    dataset_name: str,
    image_directory: str,
    split: Literal["train", "test", "val", "validation"],
    cache_dir: str = ".cache",
    image_column: str = "image",
    image_decode: bool = False,
) -> datasets.Dataset:
    """Return a HF Dataset with at least 'image' (PIL.Image) and 'image_id'."""

    name = dataset_name.lower()

    if name in ("imagenet", "imagenet1k", "in1k", "imagenet-1k"):
        hf_split = "train" if split == "train" else "validation"
        ds = datasets.load_dataset(
            "parquet",
            data_dir=image_directory,
            split=hf_split,
            cache_dir=cache_dir,
        )
        return ensure_image_id(ds, image_decode=image_decode, image_column=image_column)

    if name in ("things",):
        root = Path(image_directory)
        if split == "train":
            pattern = str(root / "training_images" / "**" / "*.jpg")
            split_key = "train"
        else:
            pattern = str(root / "test_images" / "**" / "*.jpg")
            split_key = "test"
        ds = datasets.load_dataset(
            "imagefolder",
            data_files={split_key: pattern},
            split=split_key,
            cache_dir=cache_dir,
        )
        return ensure_image_id(ds, image_decode=image_decode, image_column=image_column)

    raise ValueError(f"Invalid dataset name {name}")


def find_embedding_parquets(
    *,
    embedding_directory: str,
    dataset_name: str,
    split: Literal["train", "test", "validation", "val"],
    model_key: str,
) -> List[str]:
    """Find embedding parquet shards for one model key.

    Prefer canonical shards:
      {dataset_key}_{split}_{model_key}-part-*-of-*.parquet
    Otherwise accept rank-style:
      {dataset_key}_{split}_{model_key}.rank-*-of-*.part-*.parquet
    """
    mk = resolve_model_key(model_key)[0]
    base = f"{dataset_name}_{split}_{mk}"
    d = Path(embedding_directory)

    canon = sorted(glob.glob(str(d / f"{base}-part-*-of-*.parquet")))
    if canon:
        return canon

    rankish = sorted(glob.glob(str(d / f"{base}.rank-*-of-*.part-*.parquet")))
    return rankish


def load_embedding_dataset(
    *,
    embedding_directory: str,
    dataset_name: str,
    split: Literal["train", "test", "validation", "val"],
    model_key: str,
    cache_dir: str = ".cache",
    emb_column: str = "emb",
) -> datasets.Dataset:
    """Load one embedding dataset with at least ['image_id', emb_column]."""
    resolved_model_key = resolve_model_key(model_key)[0]
    files = find_embedding_parquets(
        embedding_directory=embedding_directory,
        dataset_name=dataset_name,
        split=split,
        model_key=resolved_model_key,
    )
    if not files:
        raise FileNotFoundError(
            f"No embedding parquet found: {dataset_name}_{split}_{resolved_model_key} in {embedding_directory}"
        )

    ds = datasets.load_dataset(
        "parquet",
        data_files={"data": files},
        split="data",
        cache_dir=cache_dir,
    )
    if "image_id" not in ds.column_names:
        raise KeyError(
            f"Embedding parquet missing 'image_id'. Example file: {files[0]}"
        )
    if emb_column not in ds.column_names:
        raise KeyError(
            f"Embedding parquet missing '{emb_column}'. Example file: {files[0]}"
        )

    keep = ["image_id", emb_column]
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    logger.info("Embedding dataset loaded.")

    return ds


def merge_datasets_by_image_id(
    primary_ds: datasets.Dataset,
    secondary_ds: datasets.Dataset,
    *,
    image_id_column: str = "image_id",
    secondary_columns: Iterable[str] = None,
    rename_map: Dict[str, str] = None,
    strict: bool = True,
    is_main_process: Optional[bool] = False,
):
    if secondary_columns is None:
        secondary_columns = [
            c for c in secondary_ds.column_names if c != image_id_column
        ]
    _rename_map = {c: c for c in secondary_columns}
    if rename_map is None:
        rename_map = _rename_map
    else:
        _rename_map.update(rename_map)
        rename_map = _rename_map

    # build lookup once
    secondary_lookup = {}
    for i in tqdm(
        range(len(secondary_ds)),
        desc="Building merging lookups ...",
        disable=not is_main_process,
    ):
        row = secondary_ds[i]
        image_id = row[image_id_column]
        if image_id in secondary_lookup:
            raise ValueError(f"Duplicate image_id found in secondary_ds: {image_id}")
        secondary_lookup[image_id] = row

    image_ids = primary_ds[image_id_column]

    merged_columns = {rename_map[c]: [] for c in secondary_columns}

    missing = []
    for image_id in image_ids:
        row = secondary_lookup.get(image_id)
        if row is None:
            if strict:
                missing.append(image_id)
                for c in secondary_columns:
                    merged_columns[rename_map[c]].append(None)
            else:
                for c in secondary_columns:
                    merged_columns[rename_map[c]].append(None)
        else:
            for c in secondary_columns:
                merged_columns[rename_map[c]].append(row[c])

    if strict and missing:
        preview = missing[:8]
        raise KeyError(
            f"Missing {image_id_column} in secondary_ds for "
            f"{len(missing)} items, e.g. {preview}"
        )

    out = primary_ds
    for new_col, values in merged_columns.items():
        out = out.add_column(new_col, values)
    return out


def parse_selected_channels(
    selected_channels: Optional[Union[str, Sequence[str]]],
) -> Optional[List[str]]:
    if selected_channels is None:
        return None
    if isinstance(selected_channels, str):
        s = selected_channels.strip()
        if not s:
            return None
        return [x.strip() for x in s.split(",") if x.strip()]
    return [str(x).strip() for x in selected_channels if str(x).strip()]


def _selected_channel_indices(
    selected_channels: Optional[Union[str, Sequence[str]]],
    channel_name_to_index: Dict[str, int] = THINGS_EEG_CHANNEL_NAME_TO_INDEX,
) -> Optional[List[int]]:
    names = parse_selected_channels(selected_channels)
    if not names:
        return None
    missing = [n for n in names if n not in channel_name_to_index]
    if missing:
        raise KeyError(f"Unknown channel names in selected_channels: {missing}")
    return [channel_name_to_index[n] for n in names]


def load_things_brain_dataset(
    *,
    data_directory: str,
    split: Literal["train", "test"],
    subject_ids: Union[int, Sequence[int], str] = (8,),
    brain_column: Literal["eeg", "meg"] = "eeg",
    avg_trials: bool = True,
    selected_channels: Optional[Union[str, Sequence[str]]] = None,
    trial_indices_by_image: Optional[Mapping[str, Sequence[int]]] = None,
    expected_trial_count: Optional[int] = None,
) -> datasets.Dataset:
    """Build HF Dataset with columns:
    - {brain_key}: Array2D float32 [C, T]
    - image_id: string
    - subject_id: int32
    """

    if trial_indices_by_image is not None:
        if split != "test":
            raise ValueError("explicit trial selection requires split='test'")
        if not avg_trials:
            raise ValueError("explicit trial selection requires avg_trials=True")
        if not isinstance(trial_indices_by_image, Mapping):
            raise ValueError("trial_indices_by_image must be a mapping")
    if expected_trial_count is not None:
        if (
            isinstance(expected_trial_count, bool)
            or not isinstance(expected_trial_count, int)
            or expected_trial_count <= 0
        ):
            raise ValueError("expected_trial_count must be a positive integer")
        if trial_indices_by_image is None:
            raise ValueError(
                "expected_trial_count requires trial_indices_by_image"
            )

    if isinstance(subject_ids, int):
        subject_ids = (subject_ids,)
    elif isinstance(subject_ids, str):
        subject_ids = [
            subj.strip() for subj in subject_ids.split(",") if subj.isdigit()
        ]

    subject_ids = [int(x) for x in subject_ids]

    sel_idx: Optional[List[int]] = None
    if brain_column == "eeg" and selected_channels is not None:
        sel_idx = _selected_channel_indices(selected_channels)

    all_ds: List[datasets.Dataset] = []
    for sid in subject_ids:
        pt_path = Path(data_directory).joinpath(f"sub-{sid:02d}", f"{split}.pt")
        loaded = torch.load(str(pt_path), weights_only=False)
        x = torch.as_tensor(loaded[brain_column])
        imgs = np.array(loaded["img"])

        if trial_indices_by_image is not None:
            if x.ndim != 4:
                raise ValueError(
                    "explicit trial selection requires an original 4-D test tensor"
                )
            if imgs.ndim == 2:
                if imgs.shape[:2] != x.shape[:2]:
                    raise ValueError(
                        "Brain/image trial shape mismatch for explicit selection: "
                        f"{tuple(x.shape[:2])} vs {tuple(imgs.shape)} in {pt_path}"
                    )
                image_ids = []
                for image_index, row in enumerate(imgs):
                    row_ids = [Path(str(value)).stem for value in row.tolist()]
                    if len(set(row_ids)) != 1:
                        raise ValueError(
                            "Brain image trial row contains mixed image IDs at "
                            f"row {image_index} in {pt_path}"
                        )
                    image_ids.append(row_ids[0])
            elif imgs.ndim == 1 and len(imgs) == x.shape[0]:
                image_ids = [Path(str(value)).stem for value in imgs.tolist()]
            else:
                raise ValueError(
                    "explicit trial selection requires one image identity per "
                    f"4-D tensor row in {pt_path}"
                )
            if len(set(image_ids)) != len(image_ids):
                raise ValueError("Brain test image IDs must be unique")
            actual_order = tuple(trial_indices_by_image)
            expected_order = tuple(image_ids)
            if actual_order != expected_order:
                actual_ids = set(actual_order)
                expected_ids = set(expected_order)
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise ValueError(
                    "trial selection requires exact image-ID coverage; "
                    f"missing={missing}, extra={extra}"
                )

            selected_rows = []
            for image_index, image_id in enumerate(image_ids):
                raw_indices = trial_indices_by_image[image_id]
                if not isinstance(raw_indices, Sequence) or isinstance(
                    raw_indices, (str, bytes)
                ):
                    raise ValueError("trial indices must be sequences of integers")
                values = tuple(raw_indices)
                if not values:
                    raise ValueError("trial indices must be non-empty")
                if any(
                    isinstance(index, bool)
                    or not isinstance(index, (int, np.integer))
                    for index in values
                ):
                    raise ValueError("trial indices must be integers")
                indices = np.asarray(values, dtype=np.int64)
                if len(np.unique(indices)) != len(indices):
                    raise ValueError("trial indices must be unique per image")
                if np.any(indices < 0) or np.any(indices >= x.shape[1]):
                    raise ValueError("trial index is out of range")
                if (
                    expected_trial_count is not None
                    and len(indices) != expected_trial_count
                ):
                    raise ValueError(
                        "formal trial selection requires exactly "
                        f"{expected_trial_count} trials per image"
                    )
                selected_rows.append(x[image_index, indices].mean(dim=0))
            x = torch.stack(selected_rows)

        if x.ndim == 4:
            if avg_trials:
                x = x.mean(dim=1)
            else:
                x = x.reshape(-1, *x.shape[2:])
        elif x.ndim != 3:
            raise ValueError(
                f"Unexpected {brain_column} shape: {tuple(x.shape)} in {pt_path}"
            )

        if sel_idx is not None:
            x = x[:, sel_idx, :]

        if avg_trials:
            if imgs.ndim == 2:
                imgs = imgs[:, 0]
            imgs = imgs.reshape(-1)[: x.shape[0]]
        else:
            imgs = imgs.reshape(-1)

        image_ids = [Path(p).stem for p in imgs.tolist()]
        if len(image_ids) != x.shape[0]:
            raise ValueError(
                f"Brain/image mismatch: {x.shape[0]} vs {len(image_ids)} for {pt_path}"
            )

        x_np = x.float().cpu().numpy()
        c_dim, t_dim = x_np.shape[1], x_np.shape[2]

        features = datasets.Features(
            {
                brain_column: datasets.Array2D(shape=(c_dim, t_dim), dtype="float32"),
                "image_id": datasets.Value("string"),
                "subject_id": datasets.Value("int32"),
            }
        )

        ds = datasets.Dataset.from_dict(
            {
                brain_column: list(x_np),
                "image_id": image_ids,
                "subject_id": [sid] * len(image_ids),
            },
            features=features,
        )
        all_ds.append(ds)

    return datasets.concatenate_datasets(all_ds) if len(all_ds) > 1 else all_ds[0]
