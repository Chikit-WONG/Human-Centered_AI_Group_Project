#!/usr/bin/env python3
"""Run the sealed Stage 1 A40 cost protocol without score observations.

The runner loads only the exact manifest-bound full InternViT+SAMGA and
BrainRW models, preloads the locked synthetic tensors, executes the
10-warmup/50-measurement alternating protocol, and publishes attempt-bound
outputs through the current immutable job claim.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter_ns

import torch
from torch import nn
from torch.nn import functional as F

from samga_brain_rw.cost_capability import (
    Stage1CostExecutionPlan,
    load_canonical_json_document,
    load_stage1_cost_execution_plan,
    load_stage1_cost_model_manifest,
    load_stage1_cost_score_input_manifest,
    stable_regular_file_sha256,
)
from samga_brain_rw.hashing import sha256_json
from samga_brain_rw.inference_cost import (
    CostProtocol,
    RawStage1CostRecord,
    benchmark_alternating_branches,
    build_raw_stage1_cost_record,
    load_cost_protocol,
    publish_raw_cost_record_exclusive,
)
from samga_brain_rw.brainrw import create_development_directory_exclusive
from samga_brain_rw.runtime_contract import require_production_runtime
from samga_brain_rw.statistics import write_development_json_exclusive


_SUBJECT = 1
_SEED = 20260720
_CONFIG_ID = "stage1_cost_v1"


def _chunk_ranges(total: int, microbatch_size: int) -> tuple[tuple[int, int], ...]:
    """Return the only allowed ascending, contiguous, full-coverage chunks."""

    if type(total) is not int or total <= 0:
        raise ValueError("chunk total must be a positive integer")
    if type(microbatch_size) is not int or microbatch_size <= 0:
        raise ValueError("microbatch size must be a positive integer")
    return tuple(
        (start, min(total, start + microbatch_size))
        for start in range(0, total, microbatch_size)
    )


def _validate_similarity(
    value: object,
    *,
    query_count: int,
    gallery_count: int,
    branch_id: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or tuple(value.shape) != (query_count, gallery_count)
    ):
        raise RuntimeError(
            f"{branch_id} must return one float32 "
            f"[{query_count},{gallery_count}] similarity matrix"
        )
    return value


def _make_internvit_similarity_callable(
    *,
    foundation_model: nn.Module,
    task_model: nn.Module,
    eeg: torch.Tensor,
    pixels: torch.Tensor,
    subject: int,
    eeg_microbatch_size: int,
    image_microbatch_size: int,
    layer_ids: Sequence[int],
    device: torch.device,
    foundation_autocast: bool,
    pooling_helper: Callable[..., tuple[torch.Tensor, torch.Tensor]],
) -> Callable[[], torch.Tensor]:
    """Build one full InternViT feature extraction + SAMGA retrieval call."""

    if eeg.ndim != 3 or pixels.ndim != 4 or eeg.shape[0] != pixels.shape[0]:
        raise ValueError("InternViT synthetic EEG/image inputs are misaligned")
    query_count = int(eeg.shape[0])
    gallery_count = int(pixels.shape[0])
    eeg_ranges = _chunk_ranges(query_count, eeg_microbatch_size)
    image_ranges = _chunk_ranges(gallery_count, image_microbatch_size)
    subject_ids = torch.full(
        (gallery_count,),
        subject,
        dtype=torch.long,
        device=device,
    )

    def run() -> torch.Tensor:
        with torch.inference_mode():
            eeg_embeddings = torch.cat(
                [
                    task_model.encode_eeg(eeg[start:end]).float()
                    for start, end in eeg_ranges
                ],
                dim=0,
            )
            image_embeddings: list[torch.Tensor] = []
            for start, end in image_ranges:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=foundation_autocast,
                ):
                    _, layer_features = pooling_helper(
                        foundation_model,
                        pixels[start:end],
                        captured_block_outputs=tuple(layer_ids),
                    )
                encoded, _ = task_model.encode_image(
                    layer_features.float(),
                    subject_ids[start:end],
                    force_global=True,
                )
                image_embeddings.append(encoded.float())
            eeg_embeddings = F.normalize(eeg_embeddings.float(), dim=-1)
            gallery_embeddings = F.normalize(
                torch.cat(image_embeddings, dim=0).float(),
                dim=-1,
            )
            similarity = eeg_embeddings @ gallery_embeddings.T
        return _validate_similarity(
            similarity,
            query_count=query_count,
            gallery_count=gallery_count,
            branch_id="internvit",
        )

    return run


def _make_brainrw_similarity_callable(
    *,
    model: nn.Module,
    eeg: torch.Tensor,
    pixels: torch.Tensor,
    eeg_microbatch_size: int,
    image_microbatch_size: int,
) -> Callable[[], torch.Tensor]:
    """Build one full BrainRW EEG + CLIP-LoRA image retrieval call."""

    if eeg.ndim != 3 or pixels.ndim != 4 or eeg.shape[0] != pixels.shape[0]:
        raise ValueError("BrainRW synthetic EEG/image inputs are misaligned")
    query_count = int(eeg.shape[0])
    gallery_count = int(pixels.shape[0])
    eeg_ranges = _chunk_ranges(query_count, eeg_microbatch_size)
    image_ranges = _chunk_ranges(gallery_count, image_microbatch_size)

    def run() -> torch.Tensor:
        with torch.inference_mode():
            eeg_embeddings = torch.cat(
                [
                    model.encode_brain(eeg[start:end]).float()
                    for start, end in eeg_ranges
                ],
                dim=0,
            )
            gallery_embeddings = torch.cat(
                [
                    model.encode_image(pixels[start:end]).float()
                    for start, end in image_ranges
                ],
                dim=0,
            )
            similarity = eeg_embeddings @ gallery_embeddings.T
        return _validate_similarity(
            similarity,
            query_count=query_count,
            gallery_count=gallery_count,
            branch_id="brainrw",
        )

    return run


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execution-config", required=True, type=Path)
    parser.add_argument("--score-inputs", required=True, type=Path)
    parser.add_argument("--model-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--expected-execution-config-sha256", required=True)
    parser.add_argument("--expected-input-bundle-sha256", required=True)
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--device", required=True, choices=("cuda",))
    return parser


def benchmark_preloaded_real_branches(
    protocol: CostProtocol,
    branch_callables: Mapping[str, Callable[[], object]],
    *,
    clock_ns: Callable[[], int] = perf_counter_ns,
    synchronize: Callable[[], object],
) -> dict[str, object]:
    """Measure only two preloaded real-model callables under the locked protocol."""

    if not isinstance(branch_callables, Mapping) or set(branch_callables) != {
        "internvit",
        "brainrw",
    }:
        raise ValueError("cost benchmark requires exactly two real branches")
    workload = protocol.synthetic_workload
    query_count = workload.get("query_count")
    gallery_count = workload.get("gallery_count")
    if type(query_count) is not int or type(gallery_count) is not int:
        raise ValueError("cost protocol similarity shape is invalid")

    def validate_last_result(branch_id: str, value: object) -> None:
        checked = _validate_similarity(
            value,
            query_count=query_count,
            gallery_count=gallery_count,
            branch_id=branch_id,
        )
        if not bool(torch.isfinite(checked).all()):
            raise RuntimeError(
                f"{branch_id} similarity matrix contains non-finite values"
            )

    return benchmark_alternating_branches(
        protocol,
        branch_callables,
        clock_ns=clock_ns,
        synchronize=synchronize,
        result_validator=validate_last_result,
    )


def _object_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be a string-keyed object")
    return dict(value)


def _manifest_role_paths(
    branch: Mapping[str, object],
    context: str,
) -> dict[str, Path]:
    raw_files = branch.get("files")
    if not isinstance(raw_files, list):
        raise ValueError(f"{context} files must be a list")
    result: dict[str, Path] = {}
    for index, raw_file in enumerate(raw_files):
        file_entry = _object_mapping(
            raw_file,
            f"{context} file[{index}]",
        )
        role = file_entry.get("role")
        path = file_entry.get("path")
        if (
            not isinstance(role, str)
            or not isinstance(path, str)
            or role in result
        ):
            raise ValueError(f"{context} file role/path is invalid")
        result[role] = Path(path)
    return result


def _require_runtime_role_paths(
    observed: Mapping[str, Path],
    declared: Mapping[str, Path],
    context: str,
) -> None:
    for role, path in observed.items():
        if declared.get(role) != path:
            raise ValueError(f"{context} executable/model role mismatch: {role}")


def _load_module_from_bound_file(path: Path, module_name: str) -> object:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load bound Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _load_bound_internvit_feature_tools(
    declared_roles: Mapping[str, Path],
) -> tuple[
    Callable[..., tuple[torch.Tensor, torch.Tensor]],
    Callable[[Path], Mapping[str, object]],
]:
    contract_path = declared_roles["internvit_feature_contract_code"]
    extractor_path = declared_roles["internvit_feature_extractor_code"]
    existing_contract = sys.modules.get("v2_5_feature_contract")
    if existing_contract is None:
        _load_module_from_bound_file(
            contract_path,
            "v2_5_feature_contract",
        )
    elif Path(str(getattr(existing_contract, "__file__", ""))) != contract_path:
        raise ValueError("loaded InternViT feature contract path mismatch")
    extractor = _load_module_from_bound_file(
        extractor_path,
        "_stage1_cost_bound_extract_v2_5_features",
    )
    helper = getattr(extractor, "collect_block_poolings", None)
    if not callable(helper):
        raise ValueError("bound InternViT extractor lacks collect_block_poolings")
    verifier = getattr(extractor, "verify_model_directory", None)
    if not callable(verifier):
        raise ValueError("bound InternViT extractor lacks verify_model_directory")
    return helper, verifier


def _validate_internvit_architecture(model: object) -> None:
    """Require the exact pinned InternViT-6B-448px-V2.5 architecture."""

    config = getattr(model, "config", None)
    if (
        getattr(config, "hidden_size", None) != 3200
        or getattr(config, "num_hidden_layers", None) != 45
        or getattr(config, "image_size", None) != 448
        or getattr(config, "patch_size", None) != 14
    ):
        raise ValueError(
            "loaded InternViT architecture differs from the sealed "
            "3200-wide, exact-45-layer, 448px/14px model"
        )


def _require_checkpoint_identity(
    *,
    branch_id: str,
    loaded_sha256: object,
    payload: Mapping[str, object],
    parameters: Mapping[str, object],
    raw_branch_reference: Mapping[str, object],
) -> tuple[int, int]:
    subject = parameters.get("representative_subject")
    seed = parameters.get("representative_seed")
    if (
        type(subject) is not int
        or type(seed) is not int
        or payload.get("subject") != subject
        or payload.get("seed") != seed
    ):
        raise ValueError(f"{branch_id} checkpoint representative-cell mismatch")
    if loaded_sha256 != raw_branch_reference.get("checkpoint_sha256"):
        raise ValueError(f"{branch_id} loaded checkpoint SHA-256 mismatch")
    return subject, seed


def _validate_brainrw_runtime_checkpoint(
    payload: Mapping[str, object],
    runtime: object,
) -> None:
    expected_pairs = {
        "semantic_environment": dict(
            getattr(runtime, "semantic_environment")
        ),
        "semantic_environment_sha256": getattr(
            runtime,
            "semantic_environment_sha256",
        ),
        "runtime_contract": dict(getattr(runtime, "contract")),
        "runtime_contract_sha256": getattr(runtime, "contract_sha256"),
    }
    for field_name, expected in expected_pairs.items():
        if payload.get(field_name) != expected:
            label = field_name.replace("_", " ")
            raise ValueError(f"BrainRW checkpoint/runtime {label} mismatch")


def _load_internvit_runtime_models(
    *,
    branch: Mapping[str, object],
    raw_branch_reference: Mapping[str, object],
    device: torch.device,
    runtime_environment_binding: Mapping[str, object],
) -> tuple[
    nn.Module,
    nn.Module,
    int,
    Callable[..., tuple[torch.Tensor, torch.Tensor]],
]:
    """Load and verify the exact pinned InternViT + Stage-0 SAMGA models."""

    parameters = _object_mapping(
        branch.get("parameters"),
        "InternViT model parameters",
    )
    checkpoint_path = Path(str(parameters["checkpoint_path"]))
    semantic_config_path = Path(str(parameters["semantic_config_path"]))
    foundation_model_path = Path(str(parameters["foundation_model_path"]))

    import train as samga_train
    from samga_brain_rw import (
        adapters,
        checkpoint_identity,
        checkpoint_io,
        checkpoints,
        feature_transforms,
        model as samga_model,
        trainer,
        upstream_samga,
    )
    from samga_brain_rw.runtime_contract import validate_environment_binding
    from transformers import AutoModel

    declared_roles = _manifest_role_paths(branch, "InternViT model")
    _require_runtime_role_paths(
        {
            "samga_adapters_code": Path(str(adapters.__file__)),
            "samga_checkpoint_identity_code": Path(
                str(checkpoint_identity.__file__)
            ),
            "samga_checkpoint_io_code": Path(str(checkpoint_io.__file__)),
            "samga_checkpoint_loader_code": Path(str(samga_train.__file__)),
            "samga_checkpoints_code": Path(str(checkpoints.__file__)),
            "samga_feature_transforms_code": Path(
                str(feature_transforms.__file__)
            ),
            "samga_model_code": Path(str(samga_model.__file__)),
            "samga_trainer_code": Path(str(trainer.__file__)),
            "samga_upstream_loader_code": Path(str(upstream_samga.__file__)),
            "samga_checkpoint": checkpoint_path,
            "samga_checkpoint_sidecar": checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".meta.json"
            ),
            "semantic_config": semantic_config_path,
        },
        declared_roles,
        "InternViT/SAMGA",
    )
    pooling_helper, model_verifier = _load_bound_internvit_feature_tools(
        declared_roles
    )
    preflight = samga_train.preflight_upstream_config(semantic_config_path)
    _require_runtime_role_paths(
        {
            "upstream_eeg_encoder_code": (
                preflight.upstream_root / "module/eeg_encoder/model.py"
            ),
            "upstream_projector_code": (
                preflight.upstream_root / "module/projector.py"
            ),
            "upstream_loss_code": preflight.upstream_root / "module/loss.py",
        },
        declared_roles,
        "pinned upstream SAMGA",
    )
    semantic_model = _object_mapping(
        preflight.semantic_payload.get("model"),
        "InternViT semantic model",
    )
    if semantic_model.get("path") != str(foundation_model_path):
        raise ValueError("InternViT foundation path differs from semantic config")
    loaded = samga_train.load_samga_checkpoint(
        checkpoint_path,
        requested_scope="train",
    )
    payload = loaded.payload
    subject, _ = _require_checkpoint_identity(
        branch_id="internvit",
        loaded_sha256=loaded.sha256,
        payload=payload,
        parameters=parameters,
        raw_branch_reference=raw_branch_reference,
    )
    if payload.get("epoch") != 60:
        raise ValueError("SAMGA representative checkpoint must be terminal epoch 60")
    runtime_state = _object_mapping(
        payload.get("runtime_state"),
        "SAMGA checkpoint runtime state",
    )
    if (
        runtime_state.get("epoch_complete") is not True
        or runtime_state.get("next_epoch") != 61
    ):
        raise ValueError("SAMGA representative checkpoint is not terminal")
    checkpoint_environment = validate_environment_binding(
        payload.get("environment")
    )
    if checkpoint_environment != dict(runtime_environment_binding):
        raise ValueError("SAMGA checkpoint/runtime environment mismatch")
    candidate = _object_mapping(
        payload.get("candidate_spec"),
        "SAMGA checkpoint candidate",
    )
    expected_candidate = {
        "stage": "stage0",
        "config_id": "internvit_baseline_v1",
        "layernorm_config_id": "s2-layernorm-off",
        "whitening_config_id": "s2-whitening-off",
        "preprojector_config_id": "s2-preproj-shared",
        "adapter_kind": "identity",
        "adapter_rank": None,
        "adapter_lr_ratio": None,
        "whitening_payload": None,
    }
    for field_name, expected in expected_candidate.items():
        if candidate.get(field_name) != expected:
            raise ValueError(
                f"SAMGA representative checkpoint {field_name} mismatch"
            )
    if (
        candidate.get("baseline_config_sha256") != preflight.semantic_sha256
        or candidate.get("semantic_config_sha256")
        != payload.get("config_sha256")
    ):
        raise ValueError("SAMGA checkpoint/config identity mismatch")
    model_inputs = trainer._validated_model_build_inputs(
        components=preflight.components,
        stage=0,
        subject=subject,
        layernorm_config_id="s2-layernorm-off",
        whitening_config_id="s2-whitening-off",
        preprojector_config_id="s2-preproj-shared",
        adapter_kind="identity",
        adapter_rank=None,
        adapter_lr_ratio=None,
        whitening=None,
    )
    task_model = trainer.SAMGARuntimeModel(model_inputs)
    task_model.load_state_dict(payload["model_state_dict"], strict=True)
    task_model = task_model.to(device=device, dtype=torch.float32).eval()

    model_verifier(foundation_model_path)
    foundation_model = AutoModel.from_pretrained(
        foundation_model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    foundation_model.eval()
    _validate_internvit_architecture(foundation_model)
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.bfloat16
        for parameter in foundation_model.parameters()
    ):
        raise ValueError("InternViT foundation parameters must be bfloat16")
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in task_model.parameters()
    ):
        raise ValueError("SAMGA task parameters must be float32")
    return foundation_model, task_model, subject, pooling_helper


def _load_brainrw_runtime_model(
    *,
    branch: Mapping[str, object],
    raw_branch_reference: Mapping[str, object],
    device: torch.device,
) -> nn.Module:
    """Load and verify the exact representative BrainRW CLIP-LoRA model."""

    from samga_brain_rw import (
        access,
        artifacts,
        brainrw as br,
        config,
        data,
        hashing,
        runtime_contract,
    )

    parameters = _object_mapping(
        branch.get("parameters"),
        "BrainRW model parameters",
    )
    checkpoint_path = Path(str(parameters["checkpoint_path"]))
    declared_roles = _manifest_role_paths(branch, "BrainRW model")
    _require_runtime_role_paths(
        {
            "brainrw_access_code": Path(str(access.__file__)),
            "brainrw_artifacts_code": Path(str(artifacts.__file__)),
            "brainrw_checkpoint": checkpoint_path,
            "brainrw_checkpoint_sidecar": checkpoint_path.with_suffix(
                checkpoint_path.suffix + ".meta.json"
            ),
            "brainrw_config_code": Path(str(config.__file__)),
            "brainrw_data_code": Path(str(data.__file__)),
            "brainrw_factory_code": Path(str(br.__file__)),
            "brainrw_hashing_code": Path(str(hashing.__file__)),
            "brainrw_runtime_contract_code": Path(str(runtime_contract.__file__)),
        },
        declared_roles,
        "BrainRW",
    )
    loaded = br.load_brainrw_checkpoint(
        checkpoint_path,
        requested_scope="val-dev",
    )
    _require_checkpoint_identity(
        branch_id="brainrw",
        loaded_sha256=loaded.sha256,
        payload=loaded.payload,
        parameters=parameters,
        raw_branch_reference=raw_branch_reference,
    )
    payload = loaded.payload
    if (
        payload.get("scope") != "train"
        or payload.get("validation_scope") != "val-dev"
        or payload.get("observed_scopes") != ["train", "val-dev"]
        or payload.get("complete") is not True
        or payload.get("training_complete") is not True
        or payload.get("global_step") != payload.get("planned_steps")
    ):
        raise ValueError("BrainRW representative checkpoint is not terminal")
    brainrw_runtime = br.probe_brainrw_production_runtime()
    _validate_brainrw_runtime_checkpoint(payload, brainrw_runtime)
    clip_path = Path(str(payload.get("clip_path")))
    _require_runtime_role_paths(
        {
            "clip_config": clip_path / "config.json",
            "clip_preprocessor_config": clip_path / "preprocessor_config.json",
            "clip_weights": clip_path / "model.safetensors",
        },
        declared_roles,
        "BrainRW CLIP",
    )
    model, _ = br.build_model_from_checkpoint(payload)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.bfloat16
        for parameter in model.parameters()
    ):
        raise ValueError("BrainRW model parameters must be bfloat16")
    return model


def _build_synthetic_inputs(
    execution_plan: Stage1CostExecutionPlan,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Preload the four locked tensors using one CPU generator in sealed order."""

    if not isinstance(execution_plan, Stage1CostExecutionPlan):
        raise TypeError("synthetic inputs require a controlled execution plan")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(execution_plan.seed)
    result: dict[str, dict[str, torch.Tensor]] = {}
    for branch_id in execution_plan.branch_order:
        branch = execution_plan.branch_payload(branch_id)
        result[branch_id] = {}
        for tensor_name, shape_name, dtype_name in (
            ("eeg", "eeg_shape", "eeg_dtype"),
            ("pixels", "image_shape", "image_dtype"),
        ):
            shape = branch[shape_name]
            if not isinstance(shape, list) or any(
                type(value) is not int or value <= 0 for value in shape
            ):
                raise ValueError("synthetic input shape differs from the seal")
            declared_dtype = branch[dtype_name]
            if declared_dtype == "float32":
                dtype = torch.float32
            elif declared_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                raise ValueError("synthetic input dtype differs from the seal")
            cpu_value = torch.randn(
                tuple(shape),
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
            result[branch_id][tensor_name] = cpu_value.to(
                device=device,
                dtype=dtype,
            )
    return result


def build_real_branch_callables(
    *,
    protocol: CostProtocol,
    execution_plan: Stage1CostExecutionPlan,
    model_manifest: Mapping[str, object],
    device: torch.device,
    runtime_environment_binding: Mapping[str, object],
) -> Mapping[str, Callable[[], object]]:
    """Preload both real models and return the only two timed branch callables."""

    if not isinstance(protocol, CostProtocol):
        raise TypeError("real branch factory requires a controlled cost protocol")
    if not isinstance(execution_plan, Stage1CostExecutionPlan):
        raise TypeError("real branch factory requires a controlled execution plan")
    if not isinstance(runtime_environment_binding, Mapping):
        raise TypeError("real branch factory requires a runtime environment binding")
    _validate_protocol(protocol)
    branches = _object_mapping(
        model_manifest.get("branches"),
        "cost model-manifest branches",
    )
    raw_model_reference = _object_mapping(
        model_manifest.get("raw_model_reference"),
        "raw cost model reference",
    )
    raw_branches = _object_mapping(
        raw_model_reference.get("branches"),
        "raw cost model branches",
    )
    if set(branches) != {"internvit", "brainrw"} or set(raw_branches) != {
        "internvit",
        "brainrw",
    }:
        raise ValueError("real branch factory requires the exact two models")
    synthetic = _build_synthetic_inputs(execution_plan, device)
    intern_foundation, intern_task, intern_subject, pooling_helper = (
        _load_internvit_runtime_models(
            branch=_object_mapping(branches["internvit"], "InternViT branch"),
            raw_branch_reference=_object_mapping(
                raw_branches["internvit"],
                "raw InternViT branch",
            ),
            device=device,
            runtime_environment_binding=runtime_environment_binding,
        )
    )
    brainrw_model = _load_brainrw_runtime_model(
        branch=_object_mapping(branches["brainrw"], "BrainRW branch"),
        raw_branch_reference=_object_mapping(
            raw_branches["brainrw"],
            "raw BrainRW branch",
        ),
        device=device,
    )
    intern_plan = execution_plan.branch_payload("internvit")
    layer_ids = intern_plan["feature_layer_ids"]
    if not isinstance(layer_ids, list):
        raise ValueError("InternViT execution plan lacks feature layer IDs")
    callables: dict[str, Callable[[], torch.Tensor]] = {
        "internvit": _make_internvit_similarity_callable(
            foundation_model=intern_foundation,
            task_model=intern_task,
            eeg=synthetic["internvit"]["eeg"],
            pixels=synthetic["internvit"]["pixels"],
            subject=intern_subject,
            eeg_microbatch_size=execution_plan.eeg_microbatch_size("internvit"),
            image_microbatch_size=execution_plan.image_microbatch_size(
                "internvit"
            ),
            layer_ids=tuple(layer_ids),
            device=device,
            foundation_autocast=True,
            pooling_helper=pooling_helper,
        ),
        "brainrw": _make_brainrw_similarity_callable(
            model=brainrw_model,
            eeg=synthetic["brainrw"]["eeg"],
            pixels=synthetic["brainrw"]["pixels"],
            eeg_microbatch_size=execution_plan.eeg_microbatch_size("brainrw"),
            image_microbatch_size=execution_plan.image_microbatch_size(
                "brainrw"
            ),
        ),
    }
    return callables


def _validate_protocol(protocol: CostProtocol) -> None:
    workload = protocol.synthetic_workload
    if (
        protocol.config_id != _CONFIG_ID
        or protocol.warmup_runs != 10
        or protocol.measured_runs != 50
        or protocol.mad_ratio_max != 0.05
        or workload.get("seed") != _SEED
        or workload.get("query_count") != 200
        or workload.get("gallery_count") != 200
        or workload.get("labels_present") is not False
        or workload.get("metrics_computed") is not False
    ):
        raise ValueError("Stage 1 cost protocol semantics drifted from the seal")


def _crossbind_representative_models(
    *,
    execution_plan: Stage1CostExecutionPlan,
    score_document: Mapping[str, object],
    model_document: Mapping[str, object],
) -> None:
    scores = score_document.get("score_inputs")
    if not isinstance(scores, list) or len(scores) != 6:
        raise ValueError("representative model binding requires six score cells")
    representative = _object_mapping(scores[0], "representative score cell")
    if (
        representative.get("subject") != 1
        or representative.get("seed") != 42
    ):
        raise ValueError("representative score cell must be sub-01/seed-42")
    branches = _object_mapping(
        model_document.get("branches"),
        "representative model branches",
    )
    raw_reference = _object_mapping(
        model_document.get("raw_model_reference"),
        "representative raw model reference",
    )
    raw_branches = _object_mapping(
        raw_reference.get("branches"),
        "representative raw model branches",
    )
    for branch_id in execution_plan.branch_order:
        plan = execution_plan.branch_payload(branch_id)
        parameters = _object_mapping(
            _object_mapping(
                branches[branch_id],
                f"{branch_id} representative model",
            ).get("parameters"),
            f"{branch_id} representative parameters",
        )
        if (
            parameters.get("representative_subject")
            != plan["representative_subject"]
            or parameters.get("representative_seed")
            != plan["representative_seed"]
        ):
            raise ValueError(f"{branch_id} representative coordinate mismatch")
        score_branch = _object_mapping(
            representative.get(branch_id),
            f"{branch_id} representative score binding",
        )
        raw_branch = _object_mapping(
            raw_branches.get(branch_id),
            f"{branch_id} representative raw model",
        )
        if raw_branch.get("checkpoint_sha256") != score_branch.get(
            "checkpoint_sha256"
        ):
            raise ValueError(
                f"{branch_id} representative checkpoint/score mismatch"
            )


def _build_runtime_reference(runtime: object) -> dict[str, object]:
    environment_binding = _object_mapping(
        getattr(runtime, "environment_binding", None),
        "production environment binding",
    )
    semantic = _object_mapping(
        environment_binding.get("semantic_environment"),
        "production semantic environment",
    )
    device = getattr(runtime, "device", None)
    if device != torch.device("cuda:0") or torch.cuda.device_count() != 1:
        raise RuntimeError("Stage 1 cost requires exactly one visible CUDA:0")
    properties = torch.cuda.get_device_properties(0)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Stage 1 cost requires A40 bfloat16 support")
    declared_environment = {
        "cuda_version": str(semantic["cuda"]),
        "python_version": str(semantic["python"]),
        "torch_version": str(semantic["torch"]),
    }
    declared_contract = {
        "accelerator": "NVIDIA A40",
        "branch_device_binding": "same_cuda_device",
        "device_index": 0,
        "device_type": "cuda",
        "process_mode": "single_process",
        "schema_version": 1,
    }
    observation = {
        "accelerator_name": str(properties.name),
        "bf16_supported": True,
        "cuda_available": True,
        "cuda_capability": [
            int(properties.major),
            int(properties.minor),
        ],
        "cuda_device_count": 1,
        "cuda_device_index": 0,
        "cuda_version": str(torch.version.cuda),
        "schema_version": 1,
        "torch_version": str(torch.__version__),
        "total_memory_bytes": int(properties.total_memory),
    }
    return {
        "declared_runtime_contract": declared_contract,
        "declared_runtime_contract_sha256": sha256_json(declared_contract),
        "declared_runtime_observation": observation,
        "declared_runtime_observation_sha256": sha256_json(observation),
        "declared_semantic_environment": declared_environment,
        "declared_semantic_environment_sha256": sha256_json(
            declared_environment
        ),
    }


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing required environment variable {name}")
    return value


def _build_job_claim_reference(run_key: str) -> dict[str, object]:
    claim_path = Path(_required_environment("SAMGA_JOB_CLAIM"))
    document, claim_file_sha256 = load_canonical_json_document(claim_path)
    if (
        set(document)
        != {"payload", "payload_sha256", "payload_type", "schema_version"}
        or document["payload_type"] != "samga_brain_rw.job_claim"
        or document["schema_version"] != 1
    ):
        raise ValueError("sealed job claim document identity mismatch")
    payload = _object_mapping(document["payload"], "sealed job claim payload")
    if (
        set(payload)
        != {
            "array_index",
            "generation",
            "job_map_sha256",
            "recovered_from_claim_sha256",
            "recovery_record_sha256",
            "row_sha256",
        }
        or document["payload_sha256"] != sha256_json(payload)
        or payload["job_map_sha256"]
        != _required_environment("SAMGA_JOB_MAP_SHA256")
        or payload["row_sha256"]
        != _required_environment("SAMGA_JOB_ROW_SHA256")
        or payload["array_index"] != 0
        or _required_environment("SAMGA_JOB_ARRAY_INDEX") != "0"
    ):
        raise ValueError("sealed job claim/job-map binding mismatch")
    generation = payload["generation"]
    if type(generation) is not int or generation <= 0:
        raise ValueError("sealed job claim generation is invalid")
    expected_parent = f"generation-{generation:06d}"
    if claim_path.name != "claim.json" or claim_path.parent.name != expected_parent:
        raise ValueError("sealed job claim path/generation mismatch")
    array_job_id = _required_environment("SLURM_ARRAY_JOB_ID")
    array_task_id = _required_environment("SLURM_ARRAY_TASK_ID")
    if array_task_id != "0" or not array_job_id.isascii() or not array_job_id.isdecimal():
        raise ValueError("SLURM array identity differs from cost row zero")
    scheduler_job_id = f"{array_job_id}_{array_task_id}"
    partition = _required_environment("SLURM_JOB_PARTITION")
    if partition != "i64m1tga40u":
        raise ValueError("Stage 1 cost claim requires i64m1tga40u")
    execution_path = Path(_required_environment("SAMGA_JOB_EXECUTION"))
    if execution_path != claim_path.with_name("execution.json"):
        raise ValueError("cost execution authority path differs from claim")
    execution_document, execution_file_sha256 = (
        load_canonical_json_document(execution_path)
    )
    if (
        execution_file_sha256
        != _required_environment("SAMGA_JOB_EXECUTION_SHA256")
        or set(execution_document)
        != {"payload", "payload_sha256", "payload_type", "schema_version"}
        or execution_document["payload_type"]
        != "samga_brain_rw.cost_execution_authority"
        or execution_document["schema_version"] != 1
    ):
        raise ValueError("cost execution authority document identity mismatch")
    execution_payload = _object_mapping(
        execution_document["payload"],
        "cost execution authority payload",
    )
    if (
        set(execution_payload)
        != {
            "array_index",
            "attempt_payload_sha256",
            "attempt_record_sha256",
            "claim_sha256",
            "generation",
            "job_map_sha256",
            "row_sha256",
            "scheduler_job_id",
        }
        or execution_document["payload_sha256"]
        != sha256_json(execution_payload)
        or execution_payload["job_map_sha256"] != payload["job_map_sha256"]
        or execution_payload["row_sha256"] != payload["row_sha256"]
        or execution_payload["array_index"] != payload["array_index"]
        or execution_payload["generation"] != generation
        or execution_payload["claim_sha256"] != claim_file_sha256
        or execution_payload["scheduler_job_id"] != scheduler_job_id
    ):
        raise ValueError("cost execution authority/claim binding mismatch")
    attempt_file_sha256 = execution_payload["attempt_record_sha256"]
    attempt_payload_sha256 = execution_payload["attempt_payload_sha256"]
    if generation == 1:
        if attempt_file_sha256 is not None or attempt_payload_sha256 is not None:
            raise ValueError(
                "first cost execution cannot bind a recovery attempt"
            )
    else:
        attempt_path = claim_path.with_name("attempt.json")
        attempt_document, actual_attempt_file_sha256 = (
            load_canonical_json_document(attempt_path)
        )
        attempt_payload = _object_mapping(
            attempt_document.get("payload"),
            "cost recovery attempt payload",
        )
        if (
            set(attempt_document)
            != {"payload", "payload_sha256", "payload_type", "schema_version"}
            or attempt_document["payload_type"]
            != "samga_brain_rw.job_attempt"
            or attempt_document["schema_version"] != 1
            or set(attempt_payload)
            != {
                "array_index",
                "claim_sha256",
                "generation",
                "job_map_sha256",
                "row_sha256",
                "scheduler_job_id",
            }
            or attempt_document["payload_sha256"]
            != sha256_json(attempt_payload)
            or attempt_payload["job_map_sha256"] != payload["job_map_sha256"]
            or attempt_payload["row_sha256"] != payload["row_sha256"]
            or attempt_payload["array_index"] != payload["array_index"]
            or attempt_payload["generation"] != generation
            or attempt_payload["claim_sha256"] != claim_file_sha256
            or attempt_payload["scheduler_job_id"] != scheduler_job_id
            or attempt_file_sha256 != actual_attempt_file_sha256
            or attempt_payload_sha256
            != attempt_document["payload_sha256"]
        ):
            raise ValueError(
                "cost execution authority/recovery attempt binding mismatch"
            )
    attempt_index = generation - 1
    return {
        "authority_execution_file_sha256": execution_file_sha256,
        "authority_execution_payload_sha256": execution_document[
            "payload_sha256"
        ],
        "attempt_id": f"attempt-{attempt_index:04d}",
        "attempt_index": attempt_index,
        "claim_id": run_key,
        "schema_version": 1,
        "slurm_job_id": scheduler_job_id,
        "slurm_partition": partition,
        "unverified_claim_sha256": claim_file_sha256,
        "unverified_previous_record_sha256": None,
    }


def _publish_sealed_outputs(
    *,
    output_dir: Path,
    raw_record: RawStage1CostRecord,
    runtime_document: Mapping[str, object],
    run_manifest_static: Mapping[str, object],
    completion_publisher: Callable[[Mapping[str, str]], object],
) -> object:
    create_development_directory_exclusive(
        output_dir,
        context="Stage 1 cost output directory",
    )
    raw_payload = raw_record.to_payload()
    claim = _object_mapping(
        raw_payload["self_attested_job_claim_reference"],
        "raw cost job claim",
    )
    raw_path = output_dir / f"stage1-cost-{claim['attempt_id']}.json"
    runtime_path = output_dir / "runtime-manifest.json"
    run_path = output_dir / "run-manifest.json"
    publish_raw_cost_record_exclusive(raw_path, raw_record)
    write_development_json_exclusive(runtime_path, dict(runtime_document))
    raw_file_sha256 = stable_regular_file_sha256(raw_path)
    runtime_file_sha256 = stable_regular_file_sha256(runtime_path)
    run_manifest = {
        **dict(run_manifest_static),
        "raw_record_file_sha256": raw_file_sha256,
        "raw_record_path": str(raw_path),
        "raw_record_sha256": raw_record.record_sha256,
        "runtime_evidence_sha256": runtime_document[
            "runtime_evidence_sha256"
        ],
        "runtime_manifest_file_sha256": runtime_file_sha256,
        "runtime_manifest_path": str(runtime_path),
    }
    write_development_json_exclusive(run_path, run_manifest)
    output_hashes = {
        "raw_record_file_sha256": raw_file_sha256,
        "run_manifest_file_sha256": stable_regular_file_sha256(run_path),
        "runtime_manifest_file_sha256": runtime_file_sha256,
    }
    return completion_publisher(output_hashes)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if (
        arguments.subject != _SUBJECT
        or arguments.seed != _SEED
        or arguments.config_id != _CONFIG_ID
    ):
        raise ValueError("Stage 1 cost runner fixed identity mismatch")

    protocol = load_cost_protocol(arguments.config)
    _validate_protocol(protocol)
    if protocol.sha256 != arguments.expected_config_sha256:
        raise ValueError("Stage 1 cost protocol SHA-256 mismatch")
    execution = load_stage1_cost_execution_plan(arguments.execution_config)
    if execution.sha256 != arguments.expected_execution_config_sha256:
        raise ValueError("Stage 1 cost execution-plan SHA-256 mismatch")

    # A live, pinned A40 runtime is mandatory before opening model or score
    # identity manifests.  This currently succeeds only inside the sealed job.
    runtime = require_production_runtime(arguments.device)
    score_document, score_file_sha256 = (
        load_stage1_cost_score_input_manifest(arguments.score_inputs)
    )
    model_document, model_file_sha256 = load_stage1_cost_model_manifest(
        arguments.model_manifest
    )
    if len(score_document["score_inputs"]) != 6:
        raise ValueError("Stage 1 cost runner requires six score identities")
    _crossbind_representative_models(
        execution_plan=execution,
        score_document=score_document,
        model_document=model_document,
    )
    expected_runner_path = (
        arguments.project_root
        / "experiments/samga_brain_rw/scripts/run_stage1_cost.py"
    )
    if Path(__file__) != expected_runner_path:
        raise ValueError("Stage 1 cost runner path differs from the sealed path")
    runner_file_sha256 = stable_regular_file_sha256(expected_runner_path)
    input_bundle_sha256 = sha256_json(
        {
            "execution_config_sha256": execution.sha256,
            "model_manifest_file_sha256": model_file_sha256,
            "runner_file_sha256": runner_file_sha256,
            "score_inputs_file_sha256": score_file_sha256,
        }
    )
    if input_bundle_sha256 != arguments.expected_input_bundle_sha256:
        raise ValueError("Stage 1 cost runner input-bundle SHA-256 mismatch")

    # No output directory or partial artifact is created before this boundary.
    # Once the execution plan is preregistered, the returned callables feed
    # benchmark_preloaded_real_branches verbatim.
    branch_callables = build_real_branch_callables(
        protocol=protocol,
        execution_plan=execution,
        model_manifest=model_document,
        device=runtime.device,
        runtime_environment_binding=runtime.environment_binding,
    )
    # Rehash every bound checkpoint, sidecar, model shard, and executable after
    # model construction and before the first synchronization/timing boundary.
    reloaded_model_document, reloaded_model_file_sha256 = (
        load_stage1_cost_model_manifest(arguments.model_manifest)
    )
    if (
        reloaded_model_document != model_document
        or reloaded_model_file_sha256 != model_file_sha256
    ):
        raise ValueError("Stage 1 cost model files changed during construction")
    benchmark = benchmark_preloaded_real_branches(
        protocol,
        branch_callables,
        synchronize=torch.cuda.synchronize,
    )
    final_model_document, final_model_file_sha256 = (
        load_stage1_cost_model_manifest(arguments.model_manifest)
    )
    final_score_document, final_score_file_sha256 = (
        load_stage1_cost_score_input_manifest(arguments.score_inputs)
    )
    final_runner_file_sha256 = stable_regular_file_sha256(
        expected_runner_path
    )
    if (
        final_model_document != model_document
        or final_model_file_sha256 != model_file_sha256
        or final_score_document != score_document
        or final_score_file_sha256 != score_file_sha256
        or final_runner_file_sha256 != runner_file_sha256
    ):
        raise ValueError("Stage 1 cost sealed inputs changed during benchmark")

    runtime_reference = _build_runtime_reference(runtime)
    runtime_evidence = runtime_reference["declared_runtime_observation"]
    runtime_evidence_sha256 = sha256_json(runtime_evidence)
    execution_file_sha256 = stable_regular_file_sha256(
        arguments.execution_config
    )
    protocol_file_sha256 = stable_regular_file_sha256(arguments.config)
    job_claim_reference = _build_job_claim_reference(arguments.run_key)
    raw_record = build_raw_stage1_cost_record(
        protocol,
        benchmark,
        runtime_reference=runtime_reference,
        model_reference=model_document["raw_model_reference"],
        job_claim_reference=job_claim_reference,
        input_reference=score_document["raw_input_reference"],
    )
    runtime_document = {
        "artifact_type": "samga_brain_rw.stage1_cost_runtime_manifest",
        "execution_config_file_sha256": execution_file_sha256,
        "execution_config_sha256": execution.sha256,
        "runtime_evidence": runtime_evidence,
        "runtime_evidence_sha256": runtime_evidence_sha256,
        "runtime_reference": runtime_reference,
        "runtime_reference_sha256": sha256_json(runtime_reference),
        "schema_version": 1,
        "scope": "stage1-cost",
    }
    run_manifest_static = {
        "artifact_type": "samga_brain_rw.stage1_cost_run_manifest",
        "authority_execution_file_sha256": job_claim_reference[
            "authority_execution_file_sha256"
        ],
        "authority_execution_path": _required_environment(
            "SAMGA_JOB_EXECUTION"
        ),
        "authority_execution_payload_sha256": job_claim_reference[
            "authority_execution_payload_sha256"
        ],
        "execution_config_file_sha256": execution_file_sha256,
        "execution_config_path": str(arguments.execution_config),
        "execution_config_sha256": execution.sha256,
        "input_bundle_sha256": input_bundle_sha256,
        "model_manifest_file_sha256": model_file_sha256,
        "model_manifest_path": str(arguments.model_manifest),
        "protocol_file_sha256": protocol_file_sha256,
        "protocol_path": str(arguments.config),
        "protocol_sha256": protocol.sha256,
        "runner_file_sha256": runner_file_sha256,
        "runner_path": str(expected_runner_path),
        "schema_version": 1,
        "scope": "stage1-cost",
        "score_inputs_file_sha256": score_file_sha256,
        "score_inputs_path": str(arguments.score_inputs),
    }
    job_maps = importlib.import_module("build_job_map")
    completion_publisher = getattr(
        job_maps,
        "complete_job_row_from_environment",
        None,
    )
    if not callable(completion_publisher):
        raise RuntimeError("job-map authority lacks completion publication")
    _publish_sealed_outputs(
        output_dir=arguments.output_dir,
        raw_record=raw_record,
        runtime_document=runtime_document,
        run_manifest_static=run_manifest_static,
        completion_publisher=completion_publisher,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
