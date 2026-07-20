"""Fail-closed semantic environment and A40 runtime bindings."""

from __future__ import annotations

import copy
import importlib.metadata
import os
import platform
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .hashing import sha256_json


PINNED_SEMANTIC_ENVIRONMENT: dict[str, object] = {
    "schema_version": 1,
    "python": "3.10.18",
    "torch": "2.10.0+cu126",
    "transformers": "4.57.6",
    "peft": "0.18.1",
    "numpy": "1.26.4",
    "scipy": "1.15.3",
    "cuda": "12.6",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
PRODUCTION_RUNTIME_CONTRACT: dict[str, object] = {
    "schema_version": 1,
    "device_type": "cuda",
    "device": "cuda:0",
    "accelerator_name": "NVIDIA A40",
    "compute_capability": [8, 6],
    "compute_dtype": "float32",
    "autocast": "disabled",
    "cudnn_sdp_enabled": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "flash_sdp_enabled": False,
    "math_sdp_enabled": True,
    "mem_efficient_sdp_enabled": False,
    "attention_evidence_scope": "torch_sdpa_policy_and_cuda0_float32_canary_only",
    "torch_sdpa_policy": "math_only",
    "torch_sdpa_canary_passed": True,
}

_SEMANTIC_ENVIRONMENT_KEYS = frozenset(PINNED_SEMANTIC_ENVIRONMENT)
_RUNTIME_CONTRACT_KEYS = frozenset(PRODUCTION_RUNTIME_CONTRACT)
_ENVIRONMENT_BINDING_KEYS = frozenset(
    {
        "schema_version",
        "semantic_environment",
        "semantic_environment_sha256",
        "runtime_contract",
        "runtime_contract_sha256",
    }
)
_PROBE_KEYS = frozenset(
    {
        "accelerator_name",
        "attention_evidence_scope",
        "torch_sdpa_policy",
        "torch_sdpa_canary_passed",
        "autocast",
        "compute_capability",
        "compute_dtype",
        "cudnn_sdp_enabled",
        "cuda_available",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "device_count",
        "flash_sdp_enabled",
        "math_sdp_enabled",
        "mem_efficient_sdp_enabled",
    }
)
_PACKAGE_NAMES = ("transformers", "peft", "numpy", "scipy")
_OFFLINE_NAMES = (
    "HF_DATASETS_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
)


@dataclass(frozen=True)
class ProductionRuntime:
    """Validated production device plus its semantic and observed evidence."""

    device: torch.device
    contract: Mapping[str, object]
    environment_binding: Mapping[str, object]
    evidence: Mapping[str, object]


def _string_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a string-keyed mapping")
    return dict(value)


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} schema mismatch")


def _validate_semantic_environment_schema(
    value: object,
) -> dict[str, object]:
    environment = _string_mapping(value, "semantic environment")
    _exact_keys(
        environment,
        _SEMANTIC_ENVIRONMENT_KEYS,
        "semantic environment",
    )
    if environment["schema_version"] != 1:
        raise ValueError("semantic environment schema_version mismatch")
    for key in _SEMANTIC_ENVIRONMENT_KEYS - {"schema_version"}:
        if not isinstance(environment[key], str) or not environment[key]:
            raise ValueError(f"semantic environment {key} must be a non-empty string")
    return environment


def _validate_runtime_contract_schema(
    value: object,
) -> dict[str, object]:
    contract = _string_mapping(value, "runtime contract")
    _exact_keys(contract, _RUNTIME_CONTRACT_KEYS, "runtime contract")
    if contract["schema_version"] != 1:
        raise ValueError("runtime contract schema_version mismatch")
    for key in (
        "device_type",
        "device",
        "accelerator_name",
        "compute_dtype",
        "autocast",
        "attention_evidence_scope",
        "torch_sdpa_policy",
    ):
        if not isinstance(contract[key], str) or not contract[key]:
            raise ValueError(f"runtime contract {key} must be a non-empty string")
    capability = contract["compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(item) is not int or item < 0 for item in capability)
    ):
        raise ValueError("runtime contract compute_capability must be two integers")
    for key in (
        "cudnn_sdp_enabled",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "flash_sdp_enabled",
        "math_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "torch_sdpa_canary_passed",
    ):
        if type(contract[key]) is not bool:
            raise ValueError(f"runtime contract {key} must be boolean")
    return contract


def capture_semantic_environment(
    *,
    environ: Mapping[str, str] | None = None,
    version_lookup: Callable[[str], str] | None = None,
    python_version: str | None = None,
    torch_version: str | None = None,
    cuda_version: str | None = None,
) -> dict[str, object]:
    """Capture only versions and offline flags that affect run semantics."""

    source_environment = os.environ if environ is None else environ

    def default_lookup(package: str) -> str:
        try:
            return importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"semantic environment package is missing: {package}"
            ) from exc

    lookup = default_lookup if version_lookup is None else version_lookup
    captured = {
        "schema_version": 1,
        "python": (
            platform.python_version() if python_version is None else python_version
        ),
        "torch": (str(torch.__version__) if torch_version is None else torch_version),
        **{package: lookup(package) for package in _PACKAGE_NAMES},
        "cuda": (str(torch.version.cuda) if cuda_version is None else cuda_version),
        **{name: source_environment.get(name, "") for name in _OFFLINE_NAMES},
    }
    return _validate_semantic_environment_schema(captured)


def require_pinned_semantic_environment(
    value: object,
) -> dict[str, object]:
    """Return a canonical copy only when every pinned semantic value matches."""

    environment = _validate_semantic_environment_schema(value)
    for key, expected in PINNED_SEMANTIC_ENVIRONMENT.items():
        if environment[key] != expected:
            raise ValueError(f"semantic environment {key} mismatch")
    return copy.deepcopy(environment)


def build_environment_binding(
    semantic_environment: object,
    runtime_contract: object,
) -> dict[str, object]:
    """Bind the semantic software environment and compute runtime by hash."""

    environment = _validate_semantic_environment_schema(semantic_environment)
    contract = _validate_runtime_contract_schema(runtime_contract)
    return {
        "schema_version": 1,
        "semantic_environment": copy.deepcopy(environment),
        "semantic_environment_sha256": sha256_json(environment),
        "runtime_contract": copy.deepcopy(contract),
        "runtime_contract_sha256": sha256_json(contract),
    }


def validate_environment_binding(value: object) -> dict[str, object]:
    """Validate exact binding schema and both claimed hashes."""

    binding = _string_mapping(value, "environment binding")
    _exact_keys(
        binding,
        _ENVIRONMENT_BINDING_KEYS,
        "environment binding",
    )
    if binding["schema_version"] != 1:
        raise ValueError("environment binding schema_version mismatch")
    environment = _validate_semantic_environment_schema(binding["semantic_environment"])
    contract = _validate_runtime_contract_schema(binding["runtime_contract"])
    if binding["semantic_environment_sha256"] != sha256_json(environment):
        raise ValueError("semantic environment hash mismatch")
    if binding["runtime_contract_sha256"] != sha256_json(contract):
        raise ValueError("runtime contract hash mismatch")
    return build_environment_binding(environment, contract)


def _torch_sdpa_cuda0_float32_canary(device: torch.device) -> bool:
    if device != torch.device("cuda:0"):
        raise RuntimeError("Torch SDPA canary requires CUDA device 0")
    query = torch.arange(
        16,
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 4, 4)
    output = torch.nn.functional.scaled_dot_product_attention(
        query,
        query,
        query,
        dropout_p=0.0,
        is_causal=False,
    )
    torch.cuda.synchronize(device)
    if (
        output.device != device
        or output.dtype != torch.float32
        or output.shape != query.shape
        or not bool(torch.isfinite(output).all().item())
    ):
        return False
    return True


def _capture_runtime_probe(
    *,
    attention_canary: Callable[[torch.device], bool] | None = None,
) -> dict[str, object]:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    accelerator_name = str(torch.cuda.get_device_name(0)) if device_count > 0 else ""
    capability = list(torch.cuda.get_device_capability(0)) if device_count > 0 else []
    flash_sdp_enabled = bool(torch.backends.cuda.flash_sdp_enabled())
    mem_efficient_sdp_enabled = bool(torch.backends.cuda.mem_efficient_sdp_enabled())
    cudnn_sdp_enabled = bool(torch.backends.cuda.cudnn_sdp_enabled())
    math_sdp_enabled = bool(torch.backends.cuda.math_sdp_enabled())
    accelerated_sdp_enabled = (
        flash_sdp_enabled or mem_efficient_sdp_enabled or cudnn_sdp_enabled
    )
    canary = (
        _torch_sdpa_cuda0_float32_canary
        if attention_canary is None
        else attention_canary
    )
    try:
        canary_passed = canary(torch.device("cuda:0"))
    except Exception as exc:
        raise RuntimeError("Torch SDPA canary failed") from exc
    if canary_passed is not True:
        raise RuntimeError("Torch SDPA canary failed")
    try:
        autocast_enabled = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:  # pragma: no cover - older supported PyTorch API
        autocast_enabled = bool(torch.is_autocast_enabled())
    return {
        "accelerator_name": accelerator_name,
        "attention_evidence_scope": "torch_sdpa_policy_and_cuda0_float32_canary_only",
        "torch_sdpa_policy": "math_only"
        if math_sdp_enabled and not accelerated_sdp_enabled
        else "accelerated_or_non_math",
        "torch_sdpa_canary_passed": canary_passed,
        "autocast": "enabled" if autocast_enabled else "disabled",
        "compute_capability": capability,
        "compute_dtype": str(torch.get_default_dtype()).removeprefix("torch."),
        "cudnn_sdp_enabled": cudnn_sdp_enabled,
        "cuda_available": cuda_available,
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "device_count": device_count,
        "flash_sdp_enabled": flash_sdp_enabled,
        "math_sdp_enabled": math_sdp_enabled,
        "mem_efficient_sdp_enabled": mem_efficient_sdp_enabled,
    }


def _validate_probe(value: object) -> dict[str, object]:
    probe = _string_mapping(value, "production runtime probe")
    _exact_keys(probe, _PROBE_KEYS, "production runtime probe")
    if type(probe["cuda_available"]) is not bool:
        raise ValueError("production runtime CUDA availability is invalid")
    if type(probe["device_count"]) is not int:
        raise ValueError("production runtime CUDA device count is invalid")
    return probe


def require_production_runtime(
    device: str | torch.device,
    *,
    probe: Mapping[str, object] | None = None,
    semantic_environment: Mapping[str, object] | None = None,
    attention_canary: Callable[[torch.device], bool] | None = None,
) -> ProductionRuntime:
    """Require the locked CUDA:0 A40 float32 runtime before any data access."""

    if str(device) not in {"cuda", "cuda:0"}:
        raise ValueError("production device must be cuda or cuda:0")
    observed = _validate_probe(
        _capture_runtime_probe(attention_canary=attention_canary)
        if probe is None
        else probe
    )
    if observed["cuda_available"] is not True:
        raise RuntimeError("production runtime requires CUDA")
    if observed["device_count"] < 1:
        raise RuntimeError("production runtime requires a CUDA device")
    expected_probe = {
        "accelerator_name": PRODUCTION_RUNTIME_CONTRACT["accelerator_name"],
        "attention_evidence_scope": PRODUCTION_RUNTIME_CONTRACT[
            "attention_evidence_scope"
        ],
        "torch_sdpa_policy": PRODUCTION_RUNTIME_CONTRACT["torch_sdpa_policy"],
        "torch_sdpa_canary_passed": PRODUCTION_RUNTIME_CONTRACT[
            "torch_sdpa_canary_passed"
        ],
        "autocast": PRODUCTION_RUNTIME_CONTRACT["autocast"],
        "compute_capability": PRODUCTION_RUNTIME_CONTRACT["compute_capability"],
        "compute_dtype": PRODUCTION_RUNTIME_CONTRACT["compute_dtype"],
        "cudnn_sdp_enabled": PRODUCTION_RUNTIME_CONTRACT["cudnn_sdp_enabled"],
        "cuda_matmul_tf32": PRODUCTION_RUNTIME_CONTRACT["cuda_matmul_tf32"],
        "cudnn_tf32": PRODUCTION_RUNTIME_CONTRACT["cudnn_tf32"],
        "flash_sdp_enabled": PRODUCTION_RUNTIME_CONTRACT["flash_sdp_enabled"],
        "math_sdp_enabled": PRODUCTION_RUNTIME_CONTRACT["math_sdp_enabled"],
        "mem_efficient_sdp_enabled": PRODUCTION_RUNTIME_CONTRACT[
            "mem_efficient_sdp_enabled"
        ],
    }
    for key, expected in expected_probe.items():
        if observed[key] != expected:
            if key == "accelerator_name":
                raise RuntimeError("production runtime requires NVIDIA A40")
            raise RuntimeError(f"production runtime {key} mismatch")
    captured_environment = (
        capture_semantic_environment()
        if semantic_environment is None
        else dict(semantic_environment)
    )
    pinned_environment = require_pinned_semantic_environment(captured_environment)
    contract = copy.deepcopy(PRODUCTION_RUNTIME_CONTRACT)
    binding = build_environment_binding(pinned_environment, contract)
    return ProductionRuntime(
        device=torch.device("cuda:0"),
        contract=MappingProxyType(contract),
        environment_binding=MappingProxyType(binding),
        evidence=MappingProxyType(copy.deepcopy(observed)),
    )
