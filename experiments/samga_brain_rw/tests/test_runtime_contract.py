from __future__ import annotations

import copy

import pytest

import samga_brain_rw.runtime_contract as runtime_contract
from samga_brain_rw import runtime_contract as runtime_contract_module
from samga_brain_rw.hashing import sha256_json
from samga_brain_rw.runtime_contract import (
    PINNED_SEMANTIC_ENVIRONMENT,
    PRODUCTION_RUNTIME_CONTRACT,
    build_environment_binding,
    capture_semantic_environment,
    require_pinned_semantic_environment,
    require_production_runtime,
    validate_environment_binding,
)


def _pinned_environment() -> dict[str, object]:
    return copy.deepcopy(PINNED_SEMANTIC_ENVIRONMENT)


def _production_probe() -> dict[str, object]:
    return {
        "accelerator_name": "NVIDIA A40",
        "attention_evidence_scope": "torch_sdpa_policy_and_cuda0_float32_canary_only",
        "torch_sdpa_policy": "math_only",
        "torch_sdpa_canary_passed": True,
        "autocast": "disabled",
        "compute_capability": [8, 6],
        "compute_dtype": "float32",
        "cudnn_sdp_enabled": False,
        "cuda_available": True,
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "device_count": 1,
        "flash_sdp_enabled": False,
        "math_sdp_enabled": True,
        "mem_efficient_sdp_enabled": False,
    }


def test_semantic_environment_capture_is_exact_and_ignores_observations() -> None:
    versions = {
        "numpy": "1.26.4",
        "peft": "0.18.1",
        "scipy": "1.15.3",
        "transformers": "4.57.6",
    }
    captured = capture_semantic_environment(
        environ={
            "CUDA_VISIBLE_DEVICES": "7",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "HOSTNAME": "observational-node",
            "SLURM_JOB_ID": "12345",
            "TRANSFORMERS_OFFLINE": "1",
        },
        version_lookup=versions.__getitem__,
        python_version="3.10.18",
        torch_version="2.10.0+cu126",
        cuda_version="12.6",
    )

    assert captured == PINNED_SEMANTIC_ENVIRONMENT
    assert "CUDA_VISIBLE_DEVICES" not in captured
    assert "HOSTNAME" not in captured
    assert "SLURM_JOB_ID" not in captured


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("python", "3.10.17"),
        ("torch", "2.9.0+cu126"),
        ("transformers", "4.57.5"),
        ("peft", "0.18.0"),
        ("numpy", "2.0.0"),
        ("scipy", "1.14.0"),
        ("cuda", "12.5"),
        ("HF_DATASETS_OFFLINE", "0"),
        ("HF_HUB_OFFLINE", "0"),
        ("TRANSFORMERS_OFFLINE", "0"),
    ],
)
def test_semantic_environment_rejects_every_pinned_mismatch(
    key: str,
    value: str,
) -> None:
    environment = _pinned_environment()
    environment[key] = value

    with pytest.raises(ValueError, match=key):
        require_pinned_semantic_environment(environment)


def test_environment_binding_canonically_hashes_both_semantic_layers() -> None:
    binding = build_environment_binding(
        _pinned_environment(),
        PRODUCTION_RUNTIME_CONTRACT,
    )

    assert binding["semantic_environment_sha256"] == sha256_json(
        PINNED_SEMANTIC_ENVIRONMENT
    )
    assert binding["runtime_contract_sha256"] == sha256_json(
        PRODUCTION_RUNTIME_CONTRACT
    )
    assert validate_environment_binding(binding) == binding

    tampered = copy.deepcopy(binding)
    tampered["semantic_environment"]["numpy"] = "2.0.0"
    with pytest.raises(ValueError, match="semantic environment hash"):
        validate_environment_binding(tampered)

    extra = copy.deepcopy(binding)
    extra["hostname"] = "must-not-be-semantic"
    with pytest.raises(ValueError, match="schema"):
        validate_environment_binding(extra)


def test_production_runtime_canonicalizes_cuda_zero_and_binds_evidence() -> None:
    runtime = require_production_runtime(
        "cuda",
        probe=_production_probe(),
        semantic_environment=_pinned_environment(),
    )

    assert str(runtime.device) == "cuda:0"
    assert runtime.contract == PRODUCTION_RUNTIME_CONTRACT
    assert runtime.environment_binding == build_environment_binding(
        PINNED_SEMANTIC_ENVIRONMENT,
        PRODUCTION_RUNTIME_CONTRACT,
    )
    assert runtime.evidence["accelerator_name"] == "NVIDIA A40"
    assert runtime.evidence["compute_capability"] == [8, 6]


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda:1"])
def test_production_runtime_rejects_every_noncanonical_device(
    device: str,
) -> None:
    with pytest.raises(ValueError, match="cuda"):
        require_production_runtime(
            device,
            probe=_production_probe(),
            semantic_environment=_pinned_environment(),
        )


def test_runtime_probe_disables_accelerated_sdp_and_reads_backend_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {
        "canary": [],
        "flash": True,
        "mem_efficient": True,
        "cudnn": True,
        "math": False,
    }
    calls: list[tuple[str, bool]] = []

    def setter(name: str):
        def apply(enabled: bool) -> None:
            calls.append((name, enabled))
            state[name] = enabled

        return apply

    for name in ("flash", "mem_efficient", "cudnn", "math"):
        monkeypatch.setattr(
            runtime_contract_module.torch.backends.cuda,
            f"enable_{name}_sdp",
            setter(name),
        )
        monkeypatch.setattr(
            runtime_contract_module.torch.backends.cuda,
            f"{name}_sdp_enabled",
            lambda name=name: state[name],
        )
    monkeypatch.setattr(
        runtime_contract_module.torch.cuda,
        "is_available",
        lambda: False,
    )
    monkeypatch.setattr(
        runtime_contract_module.torch,
        "is_autocast_enabled",
        lambda *args: False,
    )

    def canary(device: object) -> bool:
        state["canary"].append(str(device))
        return True

    observed = runtime_contract._capture_runtime_probe(
        attention_canary=canary,
    )

    assert calls == [
        ("flash", False),
        ("mem_efficient", False),
        ("cudnn", False),
        ("math", True),
    ]
    assert state["canary"] == ["cuda:0"]
    assert observed["attention_evidence_scope"] == (
        "torch_sdpa_policy_and_cuda0_float32_canary_only"
    )
    assert observed["torch_sdpa_canary_passed"] is True
    assert observed["flash_sdp_enabled"] is False
    assert observed["mem_efficient_sdp_enabled"] is False
    assert observed["cudnn_sdp_enabled"] is False
    assert observed["math_sdp_enabled"] is True


@pytest.mark.parametrize("outcome", [False, RuntimeError("kernel failure")])
def test_runtime_probe_fails_closed_when_torch_sdpa_canary_fails(
    outcome: object,
) -> None:
    def canary(device: object) -> bool:
        del device
        if isinstance(outcome, BaseException):
            raise outcome
        return bool(outcome)

    with pytest.raises(RuntimeError, match="SDPA canary"):
        runtime_contract._capture_runtime_probe(attention_canary=canary)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("cuda_available", False),
        ("device_count", 0),
        ("accelerator_name", "NVIDIA A800-SXM4-80GB"),
        ("compute_capability", [8, 0]),
        ("compute_dtype", "bfloat16"),
        ("autocast", "enabled"),
        ("cuda_matmul_tf32", True),
        ("cudnn_tf32", True),
        ("torch_sdpa_policy", "accelerated"),
        ("torch_sdpa_canary_passed", False),
        ("attention_evidence_scope", "all_model_attention_backends"),
        ("flash_sdp_enabled", True),
        ("mem_efficient_sdp_enabled", True),
        ("cudnn_sdp_enabled", True),
        ("math_sdp_enabled", False),
    ],
)
def test_production_runtime_fails_closed_on_every_probe_mismatch(
    key: str,
    value: object,
) -> None:
    probe = _production_probe()
    probe[key] = value

    with pytest.raises((RuntimeError, ValueError), match="runtime|CUDA|A40"):
        require_production_runtime(
            "cuda",
            probe=probe,
            semantic_environment=_pinned_environment(),
        )
