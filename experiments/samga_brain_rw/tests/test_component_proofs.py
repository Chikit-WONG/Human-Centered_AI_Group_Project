from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from samga_brain_rw.config import SemanticConfig, make_run_key
from samga_brain_rw.hashing import ordered_ids_sha256, sha256_json
from samga_brain_rw.stage1 import (
    BRAINRW_BRANCH_ID,
    BRAINRW_CONFIG_ID,
    BRAINRW_STAGE,
    INTERNVIT_BRANCH_ID,
    INTERNVIT_CONFIG_ID,
    INTERNVIT_STAGE,
    PILOT_COORDINATES,
    ValidatedComponentRunProof,
)


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parents[1]
FUSION_CONFIG = SemanticConfig.from_path(
    EXPERIMENT_ROOT / "configs/stage1_fusion_v1.json"
)


@dataclass(frozen=True)
class _DummyProof(ValidatedComponentRunProof):
    identity: dict[str, object]

    def revalidate(self) -> None:
        return None

    def identity_payload(self) -> dict[str, object]:
        return dict(self.identity)

    @property
    def proof_sha256(self) -> str:
        return sha256_json(self.identity)


@dataclass(frozen=True)
class _DummyIssued:
    proof: ValidatedComponentRunProof
    score: object


class _FakeCommandProof(SimpleNamespace):
    pass


@dataclass
class _FakeCompletion:
    path: Path
    sha256: str
    document: dict[str, object]
    output_hashes: dict[str, str]
    revalidation_count: int = 0

    def revalidate(self) -> None:
        self.revalidation_count += 1


@dataclass(frozen=True)
class _FakeVerified:
    payload_sha256: str
    envelope_sha256: str


@dataclass(frozen=True)
class _FakeScore:
    directory: Path
    metadata: dict[str, object]
    provenance: dict[str, object]
    query_ids: tuple[str, ...]
    gallery_ids: tuple[str, ...]
    query_ids_sha256: str
    gallery_ids_sha256: str
    verified: _FakeVerified
    scope: str = "val-dev"


def _digest(label: str, *values: object) -> str:
    return sha256_json({"label": label, "values": list(values)})


def test_component_file_hash_rejects_an_in_place_read_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"stable checkpoint bytes")
    real_fstat = component_proofs.os.fstat
    calls = 0

    def drifting_fstat(descriptor: int) -> object:
        nonlocal calls
        calls += 1
        status = real_fstat(descriptor)
        if calls != 2:
            return status
        return SimpleNamespace(
            st_ctime_ns=status.st_ctime_ns,
            st_dev=status.st_dev,
            st_ino=status.st_ino,
            st_mode=status.st_mode,
            st_mtime_ns=status.st_mtime_ns + 1,
            st_size=status.st_size,
        )

    monkeypatch.setattr(component_proofs.os, "fstat", drifting_fstat)

    with pytest.raises(ValueError, match="changed while being hashed"):
        component_proofs._sha256_file(path)

    assert calls >= 2


@pytest.fixture
def locked_component_state(tmp_path: Path, monkeypatch):
    import samga_brain_rw.component_proofs as component_proofs

    root = tmp_path / "project"
    config_root = root / "experiments/samga_brain_rw/configs"
    config_root.mkdir(parents=True)
    for name in (
        "internvit_baseline_v1.json",
        "brainrw_clip_lora_v1.json",
        "stage1_fusion_v1.json",
    ):
        shutil.copyfile(
            EXPERIMENT_ROOT / "configs" / name,
            config_root / name,
        )
    fusion = SemanticConfig.from_path(config_root / "stage1_fusion_v1.json")
    recipes = {
        INTERNVIT_BRANCH_ID: SemanticConfig.from_path(
            config_root / "internvit_baseline_v1.json"
        ),
        BRAINRW_BRANCH_ID: SemanticConfig.from_path(
            config_root / "brainrw_clip_lora_v1.json"
        ),
    }
    protocol_sha256 = _digest("protocol")
    query_ids = ("q0", "q1")
    gallery_ids = ("q0", "q1", "g2")
    query_ids_sha256 = ordered_ids_sha256(query_ids)
    gallery_ids_sha256 = ordered_ids_sha256(gallery_ids)
    semantic_environment = {
        "schema_version": 1,
        "python": "3.10.18",
        "torch": "2.10.0+cu126",
    }
    semantic_environment_sha256 = sha256_json(semantic_environment)
    maps: dict[str, dict[str, object]] = {}
    completions: dict[tuple[str, int], _FakeCompletion | None] = {}
    command_proofs: dict[tuple[str, int], _FakeCommandProof] = {}
    scores: dict[tuple[str, int], _FakeScore] = {}
    validator_calls: list[tuple[str, int]] = []

    map_specs = (
        (
            INTERNVIT_BRANCH_ID,
            "stage-0-pilot-debug.json",
            component_proofs._INTERNVIT_JOB_MAP_SHA256,
            "stage-0-pilot",
            INTERNVIT_CONFIG_ID,
        ),
        (
            BRAINRW_BRANCH_ID,
            "stage-1-brainrw-pilot-emergency.json",
            component_proofs._BRAINRW_JOB_MAP_SHA256,
            "stage-1-brainrw-pilot",
            BRAINRW_CONFIG_ID,
        ),
    )
    for branch_id, map_name, map_sha256, row_stage, config_id in map_specs:
        rows: list[dict[str, object]] = []
        for array_index, (subject, seed) in enumerate(PILOT_COORDINATES):
            manifest_path = (
                root
                / "artifacts/samga_brain_rw/protocol/manifests"
                / f"sub-{subject:02d}_protocol.json"
            )
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if not manifest_path.exists():
                manifest_path.write_text(
                    json.dumps({"subject": subject}),
                    encoding="utf-8",
                )
            recipe = recipes[branch_id]
            resolved_config_sha256 = (
                _digest("resolved-config", subject, seed)
                if branch_id == INTERNVIT_BRANCH_ID
                else recipe.sha256
            )
            input_bundle_sha256 = _digest(
                "input-bundle",
                branch_id,
                subject,
            )
            terminal_stage = (
                INTERNVIT_STAGE
                if branch_id == INTERNVIT_BRANCH_ID
                else BRAINRW_STAGE
            )
            run_key = make_run_key(
                terminal_stage,
                config_id,
                subject,
                seed,
                resolved_config_sha256,
                input_bundle_sha256,
            )
            output_dir = (
                root
                / "artifacts/samga_brain_rw"
                / row_stage
                / run_key
            )
            output_dir.mkdir(parents=True)
            score_dir = output_dir / (
                "saved_checkpoint"
                if branch_id == INTERNVIT_BRANCH_ID
                else "val_dev_scores"
            )
            score_dir.mkdir()
            config_path = config_root / (
                "internvit_baseline_v1.json"
                if branch_id == INTERNVIT_BRANCH_ID
                else "brainrw_clip_lora_v1.json"
            )
            runner = (
                "run_training_cell.py"
                if branch_id == INTERNVIT_BRANCH_ID
                else "run_brainrw_cell.py"
            )
            argv = [
                "python",
                str(
                    root
                    / "experiments/samga_brain_rw/scripts"
                    / runner
                ),
                "--mode",
                "full",
                "--subject",
                str(subject),
                "--seed",
                str(seed),
                "--config",
                str(config_path),
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--project-root",
                str(root),
                "--config-id",
                config_id,
                "--expected-config-sha256",
                resolved_config_sha256,
                "--expected-input-bundle-sha256",
                input_bundle_sha256,
                "--run-key",
                run_key,
            ]
            checkpoint_path = output_dir / (
                "checkpoint_epoch060.pt"
                if branch_id == INTERNVIT_BRANCH_ID
                else "checkpoint.pt"
            )
            run_manifest_path = output_dir / "run_manifest.json"
            checkpoint_path.write_bytes(
                f"checkpoint:{branch_id}:{subject}:{seed}".encode()
            )
            run_manifest_path.write_text(
                json.dumps(
                    {
                        "run_key": run_key,
                        "semantic_environment": semantic_environment,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            checkpoint_sha256 = component_proofs._sha256_file(checkpoint_path)
            run_manifest_sha256 = component_proofs._sha256_file(
                run_manifest_path
            )
            score_payload_sha256 = _digest(
                "score-payload",
                branch_id,
                subject,
                seed,
            )
            score_envelope_sha256 = _digest(
                "score-envelope",
                branch_id,
                subject,
                seed,
            )
            manifest_sha256 = _digest("manifest", subject)
            records_sha256 = _digest("records", subject)
            role_payload_sha256 = _digest("val-dev-role", subject)
            source_manifest_sha256 = _digest("source-manifest", subject)
            source_payload_sha256 = _digest("source-payload", subject)
            source_record = {
                "manifest_sha256": manifest_sha256,
                "records_sha256": records_sha256,
                "role": "val-dev",
                "role_payload_sha256": role_payload_sha256,
                "source_manifest_sha256": source_manifest_sha256,
                "source_payload_sha256": source_payload_sha256,
            }
            if branch_id == INTERNVIT_BRANCH_ID:
                source_record["run_key"] = run_key
            source_records = [source_record]
            source_records_sha256 = sha256_json(source_records)
            git_sha = _digest("git", branch_id)[:40]
            metadata = {
                "checkpoint_sha256": checkpoint_sha256,
                "config_sha256": resolved_config_sha256,
                "git_sha": git_sha,
                "protocol_sha256": protocol_sha256,
                "query_ids_sha256": query_ids_sha256,
                "gallery_ids_sha256": gallery_ids_sha256,
                "seed": seed,
                "source_records": source_records,
                "source_records_sha256": source_records_sha256,
                "split_role": "val-dev",
                "stage": terminal_stage,
                "subject": subject,
            }
            if branch_id == BRAINRW_BRANCH_ID:
                metadata.update(
                    {
                        "training_semantic_environment": semantic_environment,
                        "training_semantic_environment_sha256": (
                            semantic_environment_sha256
                        ),
                        "evaluation_semantic_environment": semantic_environment,
                        "evaluation_semantic_environment_sha256": (
                            semantic_environment_sha256
                        ),
                    }
                )
            score = _FakeScore(
                directory=score_dir,
                metadata=metadata,
                provenance=dict(metadata),
                query_ids=query_ids,
                gallery_ids=gallery_ids,
                query_ids_sha256=query_ids_sha256,
                gallery_ids_sha256=gallery_ids_sha256,
                verified=_FakeVerified(
                    payload_sha256=score_payload_sha256,
                    envelope_sha256=score_envelope_sha256,
                ),
            )
            scores[(branch_id, array_index)] = score
            completion_hashes = {
                "final_checkpoint_sha256": checkpoint_sha256,
                "run_manifest_sha256": run_manifest_sha256,
            }
            parity_sha256 = None
            if branch_id == INTERNVIT_BRANCH_ID:
                parity_path = output_dir / "baseline_parity.json"
                parity_path.write_text(
                    json.dumps({"run_key": run_key}),
                    encoding="utf-8",
                )
                parity_sha256 = component_proofs._sha256_file(parity_path)
                completion_hashes["parity_sha256"] = parity_sha256
            else:
                completion_hashes.update(
                    {
                        "score_envelope_sha256": score_envelope_sha256,
                        "score_payload_sha256": score_payload_sha256,
                    }
                )
            completion_path = output_dir / "completion.json"
            completion_document = {
                "schema_version": 1,
                "payload_type": "test.job_completion",
                "payload_sha256": _digest("completion-payload", branch_id, array_index),
                "payload": {
                    "array_index": array_index,
                    "job_map_sha256": map_sha256,
                    "output_hashes": completion_hashes,
                },
            }
            completion = _FakeCompletion(
                path=completion_path,
                sha256=sha256_json(completion_document),
                document=completion_document,
                output_hashes=completion_hashes,
            )
            completions[(branch_id, array_index)] = completion
            row = {
                "array_index": array_index,
                "argv": argv,
                "completion_path": str(completion_path),
                "config_id": config_id,
                "config_sha256": resolved_config_sha256,
                "input_bundle_sha256": input_bundle_sha256,
                "run_key": run_key,
                "seed": seed,
                "stage": row_stage,
                "subject": subject,
            }
            rows.append(row)
            common = {
                "alignment_sha256": ordered_ids_sha256(
                    [*query_ids, *gallery_ids]
                ),
                "completion_output_hashes": completion_hashes,
                "config_path": config_path,
                "epochs": 60 if branch_id == INTERNVIT_BRANCH_ID else 25,
                "gallery_ids_sha256": gallery_ids_sha256,
                "input_bundle_sha256": input_bundle_sha256,
                "manifest_sha256": manifest_sha256,
                "output_dir": output_dir,
                "protocol_sha256": protocol_sha256,
                "query_ids_sha256": query_ids_sha256,
                "resolved_config_sha256": resolved_config_sha256,
                "run_key": run_key,
                "run_manifest": {
                    "environment": {
                        "semantic_environment": semantic_environment,
                    },
                    "semantic_environment": semantic_environment,
                },
                "schedule_sha256": (
                    component_proofs._INTERNVIT_TRAINING_SCHEDULE_SHA256
                    if branch_id == INTERNVIT_BRANCH_ID
                    else component_proofs._BRAINRW_TRAINING_SCHEDULE_SHA256
                ),
                "scope": "val-dev",
                "sealed_argv": tuple(argv),
                "semantic_environment_sha256": (
                    semantic_environment_sha256
                ),
                "source_manifest_sha256": source_manifest_sha256,
                "source_payload_sha256": source_payload_sha256,
                "source_records_sha256": source_records_sha256,
                "split_role": "val-dev",
                "static_config_sha256": recipe.sha256,
            }
            outputs = SimpleNamespace(
                final_checkpoint_path=checkpoint_path,
                checkpoint_path=checkpoint_path,
                final_checkpoint_sha256=checkpoint_sha256,
                checkpoint_sha256=checkpoint_sha256,
                run_manifest_path=run_manifest_path,
                run_manifest_sha256=run_manifest_sha256,
                score_directory=score_dir,
                score_payload_sha256=score_payload_sha256,
                score_envelope_sha256=score_envelope_sha256,
            )
            if branch_id == INTERNVIT_BRANCH_ID:
                proof = _FakeCommandProof(
                    **common,
                    candidate_spec_sha256=_digest("candidate-spec"),
                    checkpoint=SimpleNamespace(sha256=checkpoint_sha256),
                    git_sha=git_sha,
                    manifest_path=manifest_path,
                    outputs=outputs,
                    parity_sha256=parity_sha256,
                    records_sha256=records_sha256,
                    role_payload_sha256=role_payload_sha256,
                    seed=seed,
                    stage=INTERNVIT_STAGE,
                    subject=subject,
                    terminal_score=score,
                )
            else:
                proof = _FakeCommandProof(
                    **common,
                    checkpoint=SimpleNamespace(sha256=checkpoint_sha256),
                    config=SimpleNamespace(path=config_path),
                    git_sha=git_sha,
                    manifest=SimpleNamespace(
                        path=manifest_path,
                        subject=subject,
                        manifest_sha256=manifest_sha256,
                        protocol_sha256=protocol_sha256,
                        records_sha256=records_sha256,
                        source_manifest_sha256=source_manifest_sha256,
                        source_payload_sha256=source_payload_sha256,
                        val_dev_role_sha256=role_payload_sha256,
                    ),
                    outputs=outputs,
                    score_artifact=score,
                    seed=seed,
                    subject=subject,
                )
                proof.identity = {
                    "input_bundle_sha256": input_bundle_sha256,
                    "output_dir": str(output_dir),
                    "semantic_environment_sha256": (
                        semantic_environment_sha256
                    ),
                }
                del proof.input_bundle_sha256
                del proof.output_dir
                del proof.semantic_environment_sha256
            command_proofs[(branch_id, array_index)] = proof
        maps[map_name] = {
            "array_bounds": [0, 5],
            "payload_sha256": map_sha256,
            "row_count": 6,
            "rows": rows,
        }

    def load_job_map(path: Path, *, expected_sha256: str | None = None):
        payload = maps[Path(path).name]
        assert payload["payload_sha256"] == expected_sha256
        return payload

    def branch_for_payload(payload: dict[str, object]) -> str:
        return (
            INTERNVIT_BRANCH_ID
            if payload["payload_sha256"]
            == component_proofs._INTERNVIT_JOB_MAP_SHA256
            else BRAINRW_BRANCH_ID
        )

    def load_job_completion(payload, row):
        branch_id = branch_for_payload(payload)
        return completions[(branch_id, int(row["array_index"]))]

    def command_validator(branch_id: str):
        def validate(argv, *, expected_mode):
            assert expected_mode == "full"
            rows = next(
                payload["rows"]
                for payload in maps.values()
                if branch_for_payload(payload) == branch_id
            )
            index = next(
                int(row["array_index"])
                for row in rows
                if row["argv"] == list(argv)
            )
            validator_calls.append((branch_id, index))
            return command_proofs[(branch_id, index)]

        return validate

    monkeypatch.setattr(component_proofs, "load_job_map", load_job_map)
    monkeypatch.setattr(
        component_proofs,
        "load_job_completion",
        load_job_completion,
    )
    monkeypatch.setattr(
        component_proofs,
        "validate_training_command_proof",
        command_validator(INTERNVIT_BRANCH_ID),
    )
    monkeypatch.setattr(
        component_proofs,
        "validate_brainrw_command_proof",
        command_validator(BRAINRW_BRANCH_ID),
    )
    monkeypatch.setattr(
        component_proofs,
        "ValidatedTrainingRunProof",
        _FakeCommandProof,
    )
    monkeypatch.setattr(
        component_proofs,
        "ValidatedBrainRWRunProof",
        _FakeCommandProof,
    )
    monkeypatch.setattr(
        component_proofs,
        "ScoreArtifact",
        _FakeScore,
    )
    monkeypatch.setattr(
        component_proofs,
        "_load_score_directory",
        lambda directory: next(
            score
            for score in scores.values()
            if score.directory == Path(directory)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        component_proofs,
        "common_alignment_payload",
        lambda score: {
            "gallery_ids_sha256": score.gallery_ids_sha256,
            "protocol_sha256": score.metadata["protocol_sha256"],
            "query_ids_sha256": score.query_ids_sha256,
            "scope": score.scope,
            "split_role": score.metadata["split_role"],
        },
    )
    return SimpleNamespace(
        command_proofs=command_proofs,
        completions=completions,
        fusion=fusion,
        maps=maps,
        root=root,
        scores=scores,
        validator_calls=validator_calls,
    )


def test_component_proof_public_factories_are_completion_bound(
    monkeypatch,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    issued = tuple(
        _DummyIssued(
            proof=_DummyProof({"index": index}),
            score=object(),
        )
        for index in range(2 * len(PILOT_COORDINATES))
    )
    monkeypatch.setattr(
        component_proofs,
        "_issue_locked_components",
        lambda project_root, semantic_config: issued,
    )

    proofs = component_proofs.load_stage1_component_proofs(
        PROJECT_ROOT,
        FUSION_CONFIG,
    )

    assert len(proofs) == 12
    assert all(isinstance(value, ValidatedComponentRunProof) for value in proofs)
    assert proofs == tuple(value.proof for value in issued)


def test_component_proof_factory_rejects_forged_fusion_semantics(
    monkeypatch,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    payload = FUSION_CONFIG.canonical_payload()
    payload["selection"]["scope"] = "formal-test"  # type: ignore[index]
    forged = SemanticConfig(
        _canonical=json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    monkeypatch.setattr(
        component_proofs,
        "_issue_locked_components",
        lambda project_root, semantic_config: (),
    )

    with pytest.raises(ValueError, match="locked stage1_fusion_v1"):
        component_proofs.load_stage1_component_proofs(
            PROJECT_ROOT,
            forged,
        )


def test_component_proof_factory_issues_exact_twelve_current_proofs(
    locked_component_state,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    proofs = component_proofs.load_stage1_component_proofs(
        locked_component_state.root,
        locked_component_state.fusion,
    )

    assert len(proofs) == 12
    identities = [proof.identity_payload() for proof in proofs]
    assert [
        (
            identity["branch_id"],
            identity["subject"],
            identity["seed"],
        )
        for identity in identities
    ] == [
        (branch_id, subject, seed)
        for branch_id in (INTERNVIT_BRANCH_ID, BRAINRW_BRANCH_ID)
        for subject, seed in PILOT_COORDINATES
    ]
    assert all(
        proof.proof_sha256 == sha256_json(proof.identity_payload())
        for proof in proofs
    )
    assert locked_component_state.validator_calls == [
        (branch_id, index)
        for branch_id in (INTERNVIT_BRANCH_ID, BRAINRW_BRANCH_ID)
        for index in range(6)
    ]


def test_component_proof_accepts_source_records_body_only_in_score_metadata(
    locked_component_state,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    for score in locked_component_state.scores.values():
        score.provenance.pop("source_records")

    proofs = component_proofs.load_stage1_component_proofs(
        locked_component_state.root,
        locked_component_state.fusion,
    )

    assert len(proofs) == 12


def test_component_proof_accepts_frozen_completion_json_containers(
    locked_component_state,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    for completion in locked_component_state.completions.values():
        assert completion is not None
        completion.document = MappingProxyType(
            {
                **completion.document,
                "payload": MappingProxyType(
                    dict(completion.document["payload"])  # type: ignore[arg-type]
                ),
            }
        )

    proofs = component_proofs.load_stage1_component_proofs(
        locked_component_state.root,
        locked_component_state.fusion,
    )

    assert len(proofs) == 12


def test_component_proof_revalidation_requires_same_current_completion(
    locked_component_state,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    proof = component_proofs.load_stage1_component_proofs(
        locked_component_state.root,
        locked_component_state.fusion,
    )[0]
    validator_calls = list(locked_component_state.validator_calls)
    proof.revalidate()
    proof.revalidate()
    assert locked_component_state.validator_calls == validator_calls
    current = locked_component_state.completions[(INTERNVIT_BRANCH_ID, 0)]
    assert current is not None
    assert current.revalidation_count == 3

    locked_component_state.completions[(INTERNVIT_BRANCH_ID, 0)] = None
    with pytest.raises(ValueError, match="current completion"):
        proof.revalidate()


def test_component_proof_revalidation_detects_terminal_output_tampering(
    locked_component_state,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    proof = component_proofs.load_stage1_component_proofs(
        locked_component_state.root,
        locked_component_state.fusion,
    )[0]
    output_dir = Path(
        locked_component_state.maps[
            "stage-0-pilot-debug.json"
        ]["rows"][0]["completion_path"]  # type: ignore[index]
    ).parent
    (output_dir / "checkpoint_epoch060.pt").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="terminal checkpoint"):
        proof.revalidate()


def test_component_proof_revalidation_detects_score_identity_tampering(
    locked_component_state,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    proof = component_proofs.load_stage1_component_proofs(
        locked_component_state.root,
        locked_component_state.fusion,
    )[0]
    score = locked_component_state.scores[(INTERNVIT_BRANCH_ID, 0)]
    score.metadata["protocol_sha256"] = _digest("tampered-protocol")
    score.provenance["protocol_sha256"] = score.metadata["protocol_sha256"]

    with pytest.raises(ValueError, match="protocol_sha256"):
        proof.revalidate()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("sealed_argv", ("python", "forged.py"), "sealed argv"),
        ("input_bundle_sha256", _digest("wrong-input"), "input bundle"),
        ("manifest_sha256", _digest("wrong-manifest"), "manifest identity"),
        ("protocol_sha256", _digest("wrong-protocol"), "protocol identity"),
        ("schedule_sha256", _digest("wrong-schedule"), "schedule"),
        ("resolved_config_sha256", _digest("wrong-config"), "resolved config"),
    ],
)
def test_component_proof_factory_rejects_command_identity_tampering(
    locked_component_state,
    field: str,
    replacement: object,
    message: str,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    setattr(
        locked_component_state.command_proofs[(INTERNVIT_BRANCH_ID, 0)],
        field,
        replacement,
    )

    with pytest.raises(ValueError, match=message):
        component_proofs.load_stage1_component_proofs(
            locked_component_state.root,
            locked_component_state.fusion,
        )


def test_component_proof_factory_rejects_output_or_score_path_tampering(
    locked_component_state,
) -> None:
    import samga_brain_rw.component_proofs as component_proofs

    command = locked_component_state.command_proofs[(BRAINRW_BRANCH_ID, 0)]
    command.score_artifact = _FakeScore(
        **{
            **command.score_artifact.__dict__,
            "directory": locked_component_state.root / "arbitrary-score-path",
        }
    )

    with pytest.raises(ValueError, match="score directory"):
        component_proofs.load_stage1_component_proofs(
            locked_component_state.root,
            locked_component_state.fusion,
        )


def test_cells_factory_returns_exact_order_without_score_path_parameter(
    locked_component_state,
    monkeypatch,
) -> None:
    import inspect
    import samga_brain_rw.component_proofs as component_proofs

    monkeypatch.setattr(
        component_proofs,
        "Stage1ComponentBinding",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        component_proofs,
        "Stage1CompositionCell",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    cells = component_proofs.load_stage1_composition_cells(
        locked_component_state.root,
        locked_component_state.fusion,
    )

    assert [(cell.subject, cell.seed) for cell in cells] == list(
        PILOT_COORDINATES
    )
    assert all(cell.internvit.branch_id == INTERNVIT_BRANCH_ID for cell in cells)
    assert all(cell.brainrw.branch_id == BRAINRW_BRANCH_ID for cell in cells)
    assert list(
        inspect.signature(
            component_proofs.load_stage1_composition_cells
        ).parameters
    ) == ["project_root", "semantic_config"]
