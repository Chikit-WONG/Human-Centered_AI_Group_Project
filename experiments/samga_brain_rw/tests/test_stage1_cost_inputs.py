from __future__ import annotations

import importlib.util
import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from samga_brain_rw.hashing import canonical_json_bytes, sha256_json


def _load_script(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def input_builder_module(experiment_root: Path) -> ModuleType:
    return _load_script(
        experiment_root / "scripts/build_stage1_cost_inputs.py",
        "build_stage1_cost_inputs",
    )


def test_cost_input_builder_cli_accepts_only_canonical_project_root(
    input_builder_module: ModuleType,
    tmp_path: Path,
) -> None:
    parser = input_builder_module._parser()
    assert {
        action.dest
        for action in parser._actions
        if action.dest != "help"
    } == {"project_root"}

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--project-root",
                str(tmp_path),
                "--checkpoint",
                str(tmp_path / "arbitrary.pt"),
            ]
        )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class _FakeProof:
    def __init__(self, identity: dict[str, object]) -> None:
        self._identity = identity
        self.proof_sha256 = sha256_json(identity)

    def identity_payload(self) -> dict[str, object]:
        return dict(self._identity)


def _fake_cells(tmp_path: Path) -> tuple[object, ...]:
    cells = []
    for subject in (1, 5, 8):
        for seed in (42, 43):
            cell_id = f"{subject:02d}/{seed}"
            alignment_sha256 = _digest(f"{cell_id}-alignment")
            bindings = {}
            for branch_id in ("internvit", "brainrw"):
                proof_identity = {
                    "checkpoint_sha256": _digest(
                        f"{cell_id}-{branch_id}-checkpoint"
                    ),
                    "input_bundle_sha256": _digest(
                        f"{cell_id}-{branch_id}-input"
                    ),
                    "resolved_config_sha256": _digest(
                        f"{cell_id}-{branch_id}-config"
                    ),
                    "run_key": f"{branch_id}-{subject}-{seed}",
                    "run_manifest_sha256": _digest(
                        f"{cell_id}-{branch_id}-run"
                    ),
                    "score_envelope_sha256": _digest(
                        f"{cell_id}-{branch_id}-envelope"
                    ),
                    "score_payload_sha256": _digest(
                        f"{cell_id}-{branch_id}-score"
                    ),
                    "source_payload_sha256": _digest(
                        f"{cell_id}-{branch_id}-source"
                    ),
                }
                bindings[branch_id] = SimpleNamespace(
                    alignment_sha256=alignment_sha256,
                    binding_sha256=_digest(f"{cell_id}-{branch_id}-binding"),
                    checkpoint_sha256=proof_identity["checkpoint_sha256"],
                    gallery_ids_sha256=_digest(f"{cell_id}-gallery"),
                    query_ids_sha256=_digest(f"{cell_id}-query"),
                    resolved_config_sha256=proof_identity[
                        "resolved_config_sha256"
                    ],
                    run_proof=_FakeProof(proof_identity),
                    score=SimpleNamespace(
                        directory=(
                            tmp_path
                            / branch_id
                            / cell_id.replace("/", "-")
                            / "scores"
                        ),
                        gallery_ids=tuple(f"g{index}" for index in range(200)),
                        query_ids=tuple(f"q{index}" for index in range(200)),
                    ),
                    score_envelope_sha256=proof_identity[
                        "score_envelope_sha256"
                    ],
                    score_payload_sha256=proof_identity[
                        "score_payload_sha256"
                    ],
                )
            cells.append(
                SimpleNamespace(
                    alignment_sha256=alignment_sha256,
                    brainrw=bindings["brainrw"],
                    internvit=bindings["internvit"],
                    seed=seed,
                    subject=subject,
                )
            )
    return tuple(cells)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _prepare_model_tree(
    *,
    experiment_root: Path,
    project_root: Path,
    cells: tuple[object, ...],
) -> None:
    (project_root / ".git").mkdir(parents=True)
    config_root = project_root / "experiments/samga_brain_rw/configs"
    config_root.mkdir(parents=True)
    (config_root / "stage1_fusion_v1.json").write_bytes(
        (experiment_root / "configs/stage1_fusion_v1.json").read_bytes()
    )
    foundation = (project_root / "models/internvit").resolve()
    foundation.mkdir(parents=True)
    for filename in (
        "config.json",
        "configuration_intern_vit.py",
        "flash_attention.py",
        "modeling_intern_vit.py",
        "preprocessor_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ):
        (foundation / filename).write_bytes(filename.encode("utf-8"))
    upstream = (project_root / "upstream/SAMGA").resolve()
    for relative in (
        "module/eeg_encoder/model.py",
        "module/projector.py",
        "module/loss.py",
    ):
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    intern_config = json.loads(
        (experiment_root / "configs/internvit_baseline_v1.json").read_text(
            encoding="utf-8"
        )
    )
    intern_config["upstream"]["path"] = str(upstream)
    intern_config["model"]["path"] = str(foundation)
    intern_config["model"]["config_sha256"] = _file_sha256(
        foundation / "config.json"
    )
    intern_config["model"]["preprocessor_sha256"] = _file_sha256(
        foundation / "preprocessor_config.json"
    )
    intern_config["model"]["weight_sha256"] = {
        filename: _file_sha256(foundation / filename)
        for filename in (
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        )
    }
    _write_json(config_root / "internvit_baseline_v1.json", intern_config)

    clip = (project_root / "models/clip").resolve()
    clip.mkdir(parents=True)
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ):
        (clip / filename).write_bytes(filename.encode("utf-8"))
    brain_config = json.loads(
        (experiment_root / "configs/brainrw_clip_lora_v1.json").read_text(
            encoding="utf-8"
        )
    )
    brain_config["clip"]["path"] = str(clip)
    brain_config["clip"]["config_sha256"] = _file_sha256(
        clip / "config.json"
    )
    brain_config["clip"]["weights_sha256"] = _file_sha256(
        clip / "model.safetensors"
    )
    _write_json(config_root / "brainrw_clip_lora_v1.json", brain_config)

    for relative in (
        "experiments/samga_reproduction/v2_5_feature_contract.py",
        "experiments/samga_reproduction/extract_v2_5_features.py",
        "experiments/samga_brain_rw/train.py",
        "experiments/samga_brain_rw/samga_brain_rw/access.py",
        "experiments/samga_brain_rw/samga_brain_rw/adapters.py",
        "experiments/samga_brain_rw/samga_brain_rw/artifacts.py",
        "experiments/samga_brain_rw/samga_brain_rw/brainrw.py",
        "experiments/samga_brain_rw/samga_brain_rw/checkpoint_identity.py",
        "experiments/samga_brain_rw/samga_brain_rw/checkpoint_io.py",
        "experiments/samga_brain_rw/samga_brain_rw/checkpoints.py",
        "experiments/samga_brain_rw/samga_brain_rw/config.py",
        "experiments/samga_brain_rw/samga_brain_rw/data.py",
        "experiments/samga_brain_rw/samga_brain_rw/feature_transforms.py",
        "experiments/samga_brain_rw/samga_brain_rw/hashing.py",
        "experiments/samga_brain_rw/samga_brain_rw/model.py",
        "experiments/samga_brain_rw/samga_brain_rw/runtime_contract.py",
        "experiments/samga_brain_rw/samga_brain_rw/trainer.py",
        "experiments/samga_brain_rw/samga_brain_rw/upstream_samga.py",
    ):
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))

    representative = cells[0]
    for branch_id, checkpoint_name in (
        ("internvit", "checkpoint_epoch060.pt"),
        ("brainrw", "checkpoint.pt"),
    ):
        binding = getattr(representative, branch_id)
        output_dir = Path(binding.score.directory).parent
        output_dir.mkdir(parents=True)
        checkpoint = output_dir / checkpoint_name
        checkpoint.write_bytes(f"{branch_id}-checkpoint".encode("utf-8"))
        checkpoint.with_suffix(
            checkpoint.suffix + ".meta.json"
        ).write_bytes(f"{branch_id}-sidecar".encode("utf-8"))
        checkpoint_sha256 = _file_sha256(checkpoint)
        binding.checkpoint_sha256 = checkpoint_sha256
        binding.run_proof._identity["checkpoint_sha256"] = checkpoint_sha256
        binding.run_proof.proof_sha256 = sha256_json(
            binding.run_proof._identity
        )


def test_cost_input_builder_derives_exact_twelve_score_identities(
    input_builder_module: ModuleType,
    tmp_path: Path,
) -> None:
    cells = _fake_cells(tmp_path)

    document = input_builder_module._build_score_input_manifest(cells)

    assert document["scope"] == "val-dev"
    assert [
        (value["subject"], value["seed"])
        for value in document["score_inputs"]
    ] == [
        (subject, seed)
        for subject in (1, 5, 8)
        for seed in (42, 43)
    ]
    assert len(document["score_inputs"]) == 6
    assert sum(
        1
        for value in document["score_inputs"]
        for branch_id in ("internvit", "brainrw")
        if value[branch_id]["run_proof_sha256"]
    ) == 12
    assert document["raw_input_reference"]["cells"][0]["branches"][
        "internvit"
    ]["run_key"] == "internvit-1-42"
    assert "similarity" not in repr(document).lower()
    assert "top1" not in repr(document).lower()


def test_cost_input_builder_derives_exact_model_roles_and_is_idempotent(
    experiment_root: Path,
    input_builder_module: ModuleType,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    cells = _fake_cells(tmp_path / "component-runs")
    _prepare_model_tree(
        experiment_root=experiment_root,
        project_root=project_root,
        cells=cells,
    )
    calls: list[tuple[Path, str]] = []

    def load_components(root: Path, semantic_config: object) -> tuple[object, ...]:
        calls.append((root, semantic_config.sha256))
        return cells

    first = input_builder_module.build_stage1_cost_inputs(
        project_root=project_root,
        component_loader=load_components,
    )
    second = input_builder_module.build_stage1_cost_inputs(
        project_root=project_root,
        component_loader=load_components,
    )

    input_root = (
        project_root / "artifacts/samga_brain_rw/stage-1-cost-inputs"
    )
    assert first == second
    assert first["score_inputs_path"] == str(input_root / "score-inputs.json")
    assert first["model_manifest_path"] == str(
        input_root / "model-manifest.json"
    )
    assert (
        project_root
        / "artifacts/samga_brain_rw/stage-1-cost-benchmark"
    ).is_dir()
    assert len(calls) == 2
    model = json.loads(
        (input_root / "model-manifest.json").read_text(encoding="utf-8")
    )
    roles = {
        branch_id: {
            value["role"]
            for value in model["branches"][branch_id]["files"]
        }
        for branch_id in ("internvit", "brainrw")
    }
    assert roles["internvit"] == input_builder_module._INTERNVIT_FILE_ROLES
    assert roles["brainrw"] == input_builder_module._BRAINRW_FILE_ROLES
    assert model["branches"]["internvit"]["parameters"][
        "representative_subject"
    ] == 1
    assert model["branches"]["internvit"]["parameters"][
        "representative_seed"
    ] == 42
    assert model["raw_model_reference"]["branches"]["internvit"][
        "parameter_dtypes"
    ] == {"foundation": "bfloat16", "task": "float32"}
    assert model["raw_model_reference"]["branches"]["brainrw"][
        "parameter_dtypes"
    ] == {"model": "bfloat16"}


def test_cost_input_builder_refuses_to_replace_different_existing_document(
    experiment_root: Path,
    input_builder_module: ModuleType,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    cells = _fake_cells(tmp_path / "component-runs")
    _prepare_model_tree(
        experiment_root=experiment_root,
        project_root=project_root,
        cells=cells,
    )
    input_builder_module.build_stage1_cost_inputs(
        project_root=project_root,
        component_loader=lambda root, semantic_config: cells,
    )
    score_path = (
        project_root
        / "artifacts/samga_brain_rw/stage-1-cost-inputs/score-inputs.json"
    )
    score_path.write_bytes(b'{"different":true}')

    with pytest.raises(ValueError, match="differs|identical|existing"):
        input_builder_module.build_stage1_cost_inputs(
            project_root=project_root,
            component_loader=lambda root, semantic_config: cells,
        )
    assert score_path.read_bytes() == b'{"different":true}'


def test_cost_input_builder_parent_swap_cannot_redirect_publication(
    experiment_root: Path,
    input_builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    cells = _fake_cells(tmp_path / "component-runs")
    _prepare_model_tree(
        experiment_root=experiment_root,
        project_root=project_root,
        cells=cells,
    )
    output_root = (
        project_root / "artifacts/samga_brain_rw/stage-1-cost-inputs"
    )
    detached_root = output_root.with_name("detached-cost-inputs")
    redirect_root = (tmp_path / "redirected-cost-inputs").resolve()
    redirect_root.mkdir()
    real_open = os.open
    swapped = False

    def racing_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal swapped
        name = Path(os.fspath(path)).name
        if (
            not swapped
            and name == "score-inputs.json"
            and flags & os.O_CREAT
        ):
            output_root.rename(detached_root)
            output_root.symlink_to(redirect_root, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(input_builder_module.os, "open", racing_open)

    with pytest.raises(ValueError, match="symbolic|identity|changed"):
        input_builder_module.build_stage1_cost_inputs(
            project_root=project_root,
            component_loader=lambda root, semantic_config: cells,
        )

    assert swapped is True
    assert not (redirect_root / "score-inputs.json").exists()
    assert (detached_root / "score-inputs.json").is_file()


def test_cost_input_builder_cleanup_never_unlinks_replacement_leaf(
    experiment_root: Path,
    input_builder_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = (tmp_path / "project").resolve()
    cells = _fake_cells(tmp_path / "component-runs")
    _prepare_model_tree(
        experiment_root=experiment_root,
        project_root=project_root,
        cells=cells,
    )
    score_path = (
        project_root
        / "artifacts/samga_brain_rw/stage-1-cost-inputs/score-inputs.json"
    )
    replacement = b'{"replacement":"must-survive"}'
    real_write = os.write
    replaced = False

    def racing_write(descriptor: int, data: bytes) -> int:
        nonlocal replaced
        if not replaced:
            score_path.unlink()
            score_path.write_bytes(replacement)
            replaced = True
            raise OSError("simulated write failure after leaf replacement")
        return real_write(descriptor, data)

    monkeypatch.setattr(input_builder_module.os, "write", racing_write)

    with pytest.raises(OSError, match="simulated write failure"):
        input_builder_module.build_stage1_cost_inputs(
            project_root=project_root,
            component_loader=lambda root, semantic_config: cells,
        )

    assert replaced is True
    assert score_path.read_bytes() == replacement
