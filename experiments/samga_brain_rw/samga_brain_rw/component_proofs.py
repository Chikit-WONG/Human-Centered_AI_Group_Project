"""Issue completion-bound component proofs for the locked Stage 1 pilot.

There is deliberately no public constructor from a score directory, command
line, completion document, or digest.  The two public factories start at one
canonical project root, load the two pinned six-row job maps, require every
row's current immutable completion, and invoke the branch-specific sealed
command verifier before issuing a nominal
:class:`~samga_brain_rw.stage1.ValidatedComponentRunProof`.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from scripts.build_job_map import JobCompletion, load_job_completion, load_job_map
from scripts.run_brainrw_cell import (
    BRAINRW_SCHEDULE_SHA256 as _BRAINRW_TRAINING_SCHEDULE_SHA256,
)
from scripts.run_brainrw_cell import (
    ValidatedBrainRWRunProof,
    validate_brainrw_command_proof,
)
from scripts.run_training_cell import (
    ValidatedTrainingRunProof,
    validate_training_command_proof,
)

from .config import SemanticConfig
from .fusion import common_alignment_payload
from .hashing import canonical_json_bytes, ordered_ids_sha256, sha256_json
from .scores import ScoreArtifact
from .stage1 import (
    BRAINRW_BRANCH_ID,
    BRAINRW_CONFIG_ID,
    BRAINRW_EPOCHS,
    BRAINRW_RECIPE_CONFIG_SHA256,
    BRAINRW_STAGE,
    INTERNVIT_BRANCH_ID,
    INTERNVIT_CONFIG_ID,
    INTERNVIT_EPOCHS,
    INTERNVIT_RECIPE_CONFIG_SHA256,
    INTERNVIT_STAGE,
    PILOT_COORDINATES,
    STAGE1_FUSION_CONFIG_SHA256,
    VALIDATED_RUN_PROOF_TYPE,
    Stage1ComponentBinding,
    Stage1CompositionCell,
    ValidatedComponentRunProof,
)
from .trainer import SCHEDULE_SHA256 as _INTERNVIT_TRAINING_SCHEDULE_SHA256


_INTERNVIT_JOB_MAP_NAME = "stage-0-pilot-debug.json"
_INTERNVIT_JOB_MAP_SHA256 = (
    "dfc087012b8d382a912030fe3367d23b38f8f0e127c58dc860e51e2ebeb43071"
)
_BRAINRW_JOB_MAP_NAME = "stage-1-brainrw-pilot-emergency.json"
_BRAINRW_JOB_MAP_SHA256 = (
    "3ba23038b1c0bb7295418f91278b6d90a2f5bc7f3495ff033dcd873d5dc0698a"
)
_JOB_MAP_ROOT = Path("artifacts/samga_brain_rw/job_maps")
_CONFIG_ROOT = Path("experiments/samga_brain_rw/configs")
_PROTOCOL_ROOT = Path("artifacts/samga_brain_rw/protocol/manifests")
_ISSUER_TOKEN = object()

_BRANCH_SPECS = {
    INTERNVIT_BRANCH_ID: {
        "config_id": INTERNVIT_CONFIG_ID,
        "config_name": "internvit_baseline_v1.json",
        "epochs": INTERNVIT_EPOCHS,
        "job_map_name": _INTERNVIT_JOB_MAP_NAME,
        "job_map_sha256": _INTERNVIT_JOB_MAP_SHA256,
        "recipe_sha256": INTERNVIT_RECIPE_CONFIG_SHA256,
        "row_stage": "stage-0-pilot",
        "runner_name": "run_training_cell.py",
        "score_name": "saved_checkpoint",
        "stage": INTERNVIT_STAGE,
        "training_schedule_sha256": _INTERNVIT_TRAINING_SCHEDULE_SHA256,
    },
    BRAINRW_BRANCH_ID: {
        "config_id": BRAINRW_CONFIG_ID,
        "config_name": "brainrw_clip_lora_v1.json",
        "epochs": BRAINRW_EPOCHS,
        "job_map_name": _BRAINRW_JOB_MAP_NAME,
        "job_map_sha256": _BRAINRW_JOB_MAP_SHA256,
        "recipe_sha256": BRAINRW_RECIPE_CONFIG_SHA256,
        "row_stage": "stage-1-brainrw-pilot",
        "runner_name": "run_brainrw_cell.py",
        "score_name": "val_dev_scores",
        "stage": BRAINRW_STAGE,
        "training_schedule_sha256": _BRAINRW_TRAINING_SCHEDULE_SHA256,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(Path(path), flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"proof input is not a regular file: {path}")
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
    except BaseException:
        os.close(descriptor)
        raise
    with stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(stream.fileno())
        def identity(value: object) -> tuple[object, ...]:
            return (
                getattr(value, "st_dev"),
                getattr(value, "st_ino"),
                getattr(value, "st_mode"),
                getattr(value, "st_size"),
                getattr(value, "st_mtime_ns"),
                getattr(value, "st_ctime_ns"),
            )

        if identity(before) != identity(after):
            raise ValueError(f"proof input changed while being hashed: {path}")
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise ValueError(
                f"proof input changed while being hashed: {path}"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino)
            != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"proof input changed while being hashed: {path}")
    return digest.hexdigest()


def _clone_json(value: object) -> object:
    if isinstance(value, Mapping):
        native: object = {
            str(key): _clone_json(child)
            for key, child in value.items()
        }
    elif isinstance(value, (list, tuple)):
        native = [_clone_json(child) for child in value]
    else:
        native = value
    return json.loads(canonical_json_bytes(native).decode("utf-8"))


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    return value


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping")
    cloned = _clone_json(value)
    if not isinstance(cloned, dict):
        raise AssertionError(f"{context} did not clone to an object")
    return cloned


def _require_equal(actual: object, expected: object, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context} mismatch")


def _require_locked_semantic_config(
    project_root: Path,
    semantic_config: SemanticConfig,
) -> tuple[Path, SemanticConfig]:
    if not isinstance(semantic_config, SemanticConfig):
        raise TypeError("semantic_config must be a SemanticConfig")
    if semantic_config.sha256 != STAGE1_FUSION_CONFIG_SHA256:
        raise ValueError(
            "semantic_config differs from the locked stage1_fusion_v1"
        )
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be a pathlib.Path")
    root = project_root.resolve(strict=True)
    if not root.is_dir() or root != project_root:
        raise ValueError("project_root must be an absolute canonical directory")
    sealed = SemanticConfig.from_path(
        root / _CONFIG_ROOT / "stage1_fusion_v1.json"
    )
    if (
        sealed.sha256 != STAGE1_FUSION_CONFIG_SHA256
        or sealed.canonical_payload() != semantic_config.canonical_payload()
    ):
        raise ValueError(
            "project semantic config differs from locked stage1_fusion_v1"
        )
    return root, sealed


def _component_schedule(
    branch_id: str,
    recipe: SemanticConfig,
) -> dict[str, object]:
    payload = recipe.canonical_payload()
    spec = _BRANCH_SPECS[branch_id]
    common: dict[str, object] = {
        "artifact_type": "samga_brain_rw.stage1_component_schedule",
        "branch_id": branch_id,
        "config_id": spec["config_id"],
        "config_sha256": recipe.sha256,
        "epochs": spec["epochs"],
        "schema_version": 1,
    }
    if branch_id == INTERNVIT_BRANCH_ID:
        common["task"] = payload["task"]
    else:
        common["optimizer"] = payload["optimizer"]
        common["training"] = payload["training"]
    return common


def _validate_locked_map(
    payload: Mapping[str, object],
    *,
    branch_id: str,
) -> tuple[dict[str, object], ...]:
    spec = _BRANCH_SPECS[branch_id]
    _require_equal(
        payload.get("payload_sha256"),
        spec["job_map_sha256"],
        f"{branch_id} job-map hash",
    )
    _require_equal(
        payload.get("array_bounds"),
        [0, 5],
        f"{branch_id} job-map bounds",
    )
    _require_equal(
        payload.get("row_count"),
        6,
        f"{branch_id} job-map row count",
    )
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(
        raw_rows,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"{branch_id} job-map rows must be a sequence")
    rows = tuple(_mapping(row, f"{branch_id} job-map row") for row in raw_rows)
    if len(rows) != 6:
        raise ValueError(f"{branch_id} job-map must contain six rows")
    coordinates = []
    for index, row in enumerate(rows):
        coordinate = (row.get("subject"), row.get("seed"))
        coordinates.append(coordinate)
        required = {
            "array_index": index,
            "config_id": spec["config_id"],
            "stage": spec["row_stage"],
        }
        for field_name, expected in required.items():
            _require_equal(
                row.get(field_name),
                expected,
                f"{branch_id} row {field_name}",
            )
        argv = row.get("argv")
        if not isinstance(argv, list) or len(argv) < 2:
            raise ValueError(f"{branch_id} row sealed argv is invalid")
        expected_runner = str(
            Path(argv[1]).parents[1] / "scripts" / str(spec["runner_name"])
        )
        _require_equal(
            argv[1],
            expected_runner,
            f"{branch_id} row runner",
        )
    _require_equal(
        tuple(coordinates),
        PILOT_COORDINATES,
        f"{branch_id} job-map coordinates",
    )
    return rows


def _score_metadata(score: ScoreArtifact) -> dict[str, object]:
    metadata = _mapping(score.metadata, "component score metadata")
    provenance = _mapping(score.provenance, "component score provenance")
    for field_name in (
        "checkpoint_sha256",
        "config_sha256",
        "gallery_ids_sha256",
        "git_sha",
        "protocol_sha256",
        "query_ids_sha256",
        "seed",
        "source_records_sha256",
        "split_role",
        "stage",
        "subject",
    ):
        _require_equal(
            provenance.get(field_name),
            metadata.get(field_name),
            f"component score provenance {field_name}",
        )
    return metadata


def _semantic_environment(
    branch_id: str,
    command_proof: object,
    score_metadata: Mapping[str, object],
) -> dict[str, object]:
    if branch_id == INTERNVIT_BRANCH_ID:
        run_manifest = _mapping(
            getattr(command_proof, "run_manifest"),
            "InternViT run manifest",
        )
        environment = _mapping(
            run_manifest.get("environment"),
            "InternViT run environment",
        )
        semantic = _mapping(
            environment.get("semantic_environment"),
            "InternViT semantic environment",
        )
    else:
        semantic = _mapping(
            score_metadata.get("training_semantic_environment"),
            "BrainRW training semantic environment",
        )
        _require_equal(
            score_metadata.get("evaluation_semantic_environment"),
            semantic,
            "BrainRW evaluation semantic environment",
        )
    return semantic


def _command_score(
    branch_id: str,
    command_proof: object,
) -> ScoreArtifact:
    score = (
        getattr(command_proof, "terminal_score", None)
        if branch_id == INTERNVIT_BRANCH_ID
        else getattr(command_proof, "score_artifact", None)
    )
    if not isinstance(score, ScoreArtifact):
        raise TypeError(f"{branch_id} command proof lacks a ScoreArtifact")
    return score


def _command_value(
    branch_id: str,
    command_proof: object,
    field_name: str,
) -> object:
    if hasattr(command_proof, field_name):
        return getattr(command_proof, field_name)
    if branch_id == BRAINRW_BRANCH_ID:
        identity = getattr(command_proof, "identity", None)
        if isinstance(identity, Mapping) and field_name in identity:
            return identity[field_name]
    raise ValueError(f"{branch_id} command proof lacks {field_name}")


def _command_paths(
    branch_id: str,
    command_proof: object,
) -> tuple[Path, Path, Path, Path]:
    output_dir = Path(
        str(_command_value(branch_id, command_proof, "output_dir"))
    )
    if branch_id == INTERNVIT_BRANCH_ID:
        config_path = Path(getattr(command_proof, "config_path"))
        manifest_path = Path(getattr(command_proof, "manifest_path"))
        outputs = getattr(command_proof, "outputs")
        checkpoint_path = Path(outputs.final_checkpoint_path)
    else:
        config_path = Path(getattr(command_proof, "config").path)
        manifest_path = Path(getattr(command_proof, "manifest").path)
        outputs = getattr(command_proof, "outputs")
        checkpoint_path = Path(outputs.checkpoint_path)
    return output_dir, config_path, manifest_path, checkpoint_path


def _validate_command_crossbindings(
    *,
    branch_id: str,
    command_proof: object,
    completion: JobCompletion,
    project_root: Path,
    recipe: SemanticConfig,
    row: Mapping[str, object],
) -> ScoreArtifact:
    spec = _BRANCH_SPECS[branch_id]
    expected_type = (
        ValidatedTrainingRunProof
        if branch_id == INTERNVIT_BRANCH_ID
        else ValidatedBrainRWRunProof
    )
    if not isinstance(command_proof, expected_type):
        raise TypeError(f"{branch_id} validator returned the wrong proof type")
    argv = tuple(str(value) for value in row["argv"])  # type: ignore[index]
    _require_equal(
        getattr(command_proof, "sealed_argv"),
        argv,
        f"{branch_id} sealed argv",
    )
    for proof_name, row_name, label in (
        ("input_bundle_sha256", "input_bundle_sha256", "input bundle"),
        ("resolved_config_sha256", "config_sha256", "resolved config"),
        ("run_key", "run_key", "run identity"),
    ):
        _require_equal(
            _command_value(branch_id, command_proof, proof_name),
            row[row_name],
            f"{branch_id} {label}",
        )
    _require_equal(
        getattr(command_proof, "static_config_sha256"),
        recipe.sha256,
        f"{branch_id} recipe config",
    )
    _require_equal(
        getattr(command_proof, "schedule_sha256"),
        spec["training_schedule_sha256"],
        f"{branch_id} schedule",
    )
    _require_equal(
        getattr(command_proof, "epochs"),
        spec["epochs"],
        f"{branch_id} epochs",
    )
    _require_equal(
        getattr(command_proof, "scope"),
        "val-dev",
        f"{branch_id} scope",
    )
    _require_equal(
        getattr(command_proof, "split_role"),
        "val-dev",
        f"{branch_id} split role",
    )
    output_dir, config_path, manifest_path, checkpoint_path = _command_paths(
        branch_id,
        command_proof,
    )
    expected_output_dir = Path(str(row["completion_path"])).parent
    expected_config_path = (
        project_root / _CONFIG_ROOT / str(spec["config_name"])
    )
    expected_manifest_path = (
        project_root
        / _PROTOCOL_ROOT
        / f"sub-{int(row['subject']):02d}_protocol.json"
    )
    _require_equal(
        output_dir,
        expected_output_dir,
        f"{branch_id} output directory",
    )
    _require_equal(
        config_path,
        expected_config_path,
        f"{branch_id} config path",
    )
    _require_equal(
        manifest_path,
        expected_manifest_path,
        f"{branch_id} manifest path",
    )
    expected_score_dir = output_dir / str(spec["score_name"])
    score = _command_score(branch_id, command_proof)
    _require_equal(
        score.directory,
        expected_score_dir,
        f"{branch_id} score directory",
    )
    completion_hashes = dict(completion.output_hashes)
    proof_hashes = dict(getattr(command_proof, "completion_output_hashes"))
    _require_equal(
        proof_hashes,
        completion_hashes,
        f"{branch_id} completion output hashes",
    )
    outputs = getattr(command_proof, "outputs")
    _require_equal(
        _sha256_file(checkpoint_path),
        completion_hashes["final_checkpoint_sha256"],
        f"{branch_id} checkpoint output",
    )
    _require_equal(
        _sha256_file(Path(outputs.run_manifest_path)),
        completion_hashes["run_manifest_sha256"],
        f"{branch_id} run manifest output",
    )
    if branch_id == INTERNVIT_BRANCH_ID:
        _require_equal(
            _sha256_file(output_dir / "baseline_parity.json"),
            completion_hashes["parity_sha256"],
            "InternViT parity output",
        )
        _require_equal(
            getattr(command_proof, "parity_sha256"),
            completion_hashes["parity_sha256"],
            "InternViT parity proof",
        )
    else:
        _require_equal(
            score.verified.payload_sha256,
            completion_hashes["score_payload_sha256"],
            "BrainRW score payload output",
        )
        _require_equal(
            score.verified.envelope_sha256,
            completion_hashes["score_envelope_sha256"],
            "BrainRW score envelope output",
        )
    return score


def _identity_payload(
    *,
    branch_id: str,
    command_proof: object,
    completion: JobCompletion,
    recipe: SemanticConfig,
    row: Mapping[str, object],
    score: ScoreArtifact,
) -> dict[str, object]:
    spec = _BRANCH_SPECS[branch_id]
    metadata = _score_metadata(score)
    source_records = metadata["source_records"]
    if not isinstance(source_records, Sequence) or isinstance(
        source_records,
        (str, bytes, bytearray),
    ):
        raise ValueError(f"{branch_id} score source_records are invalid")
    source_records_payload = _clone_json(source_records)
    if not isinstance(source_records_payload, list) or not source_records_payload:
        raise ValueError(f"{branch_id} score source_records are empty")
    first_record = source_records_payload[0]
    if not isinstance(first_record, dict):
        raise ValueError(f"{branch_id} score source record is invalid")
    crossbindings = (
        (
            "protocol_sha256",
            metadata["protocol_sha256"],
            "protocol identity",
        ),
        (
            "manifest_sha256",
            first_record["manifest_sha256"],
            "manifest identity",
        ),
        (
            "source_manifest_sha256",
            first_record["source_manifest_sha256"],
            "source manifest identity",
        ),
        (
            "source_payload_sha256",
            first_record["source_payload_sha256"],
            "source payload identity",
        ),
        (
            "source_records_sha256",
            metadata["source_records_sha256"],
            "source-record identity",
        ),
        (
            "query_ids_sha256",
            score.query_ids_sha256,
            "query ID identity",
        ),
        (
            "gallery_ids_sha256",
            score.gallery_ids_sha256,
            "gallery ID identity",
        ),
    )
    for field_name, expected, context in crossbindings:
        _require_equal(
            getattr(command_proof, field_name),
            expected,
            f"{branch_id} {context}",
        )
    _require_equal(
        getattr(command_proof, "checkpoint").sha256,
        metadata["checkpoint_sha256"],
        f"{branch_id} checkpoint identity",
    )
    alignment = common_alignment_payload(score)
    _require_equal(
        getattr(command_proof, "alignment_sha256"),
        ordered_ids_sha256([*score.query_ids, *score.gallery_ids]),
        f"{branch_id} command alignment",
    )
    semantic_environment = _semantic_environment(
        branch_id,
        command_proof,
        metadata,
    )
    semantic_environment_sha256 = sha256_json(semantic_environment)
    _require_equal(
        _command_value(
            branch_id,
            command_proof,
            "semantic_environment_sha256",
        ),
        semantic_environment_sha256,
        f"{branch_id} semantic environment",
    )
    schedule = _component_schedule(branch_id, recipe)
    completion_hashes = dict(completion.output_hashes)
    manifest = (
        getattr(command_proof, "manifest", None)
        if branch_id == BRAINRW_BRANCH_ID
        else None
    )
    records_sha256 = (
        manifest.records_sha256
        if manifest is not None
        else getattr(command_proof, "records_sha256")
    )
    role_payload_sha256 = (
        manifest.val_dev_role_sha256
        if manifest is not None
        else getattr(command_proof, "role_payload_sha256")
    )
    git_sha = metadata["git_sha"]
    return {
        "alignment": alignment,
        "alignment_sha256": sha256_json(alignment),
        "artifact_type": VALIDATED_RUN_PROOF_TYPE,
        "branch_id": branch_id,
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "completion_output_hashes": completion_hashes,
        "completion_sha256": completion.sha256,
        "epochs": spec["epochs"],
        "gallery_ids_sha256": score.gallery_ids_sha256,
        "git_sha": git_sha,
        "input_bundle_sha256": row["input_bundle_sha256"],
        "manifest_sha256": first_record["manifest_sha256"],
        "protocol_sha256": metadata["protocol_sha256"],
        "query_ids_sha256": score.query_ids_sha256,
        "recipe_config_id": spec["config_id"],
        "recipe_config_sha256": recipe.sha256,
        "records_sha256": records_sha256,
        "resolved_config_sha256": row["config_sha256"],
        "role_payload_sha256": role_payload_sha256,
        "run_key": row["run_key"],
        "run_manifest_sha256": completion_hashes["run_manifest_sha256"],
        "schedule": schedule,
        "schedule_sha256": sha256_json(schedule),
        "schema_version": 1,
        "scope": "val-dev",
        "score_envelope_sha256": score.verified.envelope_sha256,
        "score_payload_sha256": score.verified.payload_sha256,
        "seed": row["seed"],
        "semantic_environment": semantic_environment,
        "semantic_environment_sha256": semantic_environment_sha256,
        "source_manifest_sha256": first_record["source_manifest_sha256"],
        "source_payload_sha256": first_record["source_payload_sha256"],
        "source_records": source_records_payload,
        "source_records_sha256": metadata["source_records_sha256"],
        "split_role": "val-dev",
        "stage": spec["stage"],
        "subject": row["subject"],
    }


def _load_score_directory(directory: Path) -> ScoreArtifact:
    return ScoreArtifact.load(
        directory,
        allowed_scopes={"val-dev"},
    )


def _validate_reloaded_score(
    *,
    branch_id: str,
    identity: Mapping[str, object],
    score: ScoreArtifact,
) -> None:
    if not isinstance(score, ScoreArtifact):
        raise TypeError(f"{branch_id} score reload did not return ScoreArtifact")
    metadata = _score_metadata(score)
    expected_metadata = {
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "config_sha256": identity["resolved_config_sha256"],
        "gallery_ids_sha256": identity["gallery_ids_sha256"],
        "git_sha": identity["git_sha"],
        "protocol_sha256": identity["protocol_sha256"],
        "query_ids_sha256": identity["query_ids_sha256"],
        "seed": identity["seed"],
        "source_records_sha256": identity["source_records_sha256"],
        "split_role": identity["split_role"],
        "stage": identity["stage"],
        "subject": identity["subject"],
    }
    for field_name, expected in expected_metadata.items():
        _require_equal(
            metadata.get(field_name),
            expected,
            f"{branch_id} reloaded score {field_name}",
        )
    _require_equal(
        _clone_json(metadata["source_records"]),
        _clone_json(identity["source_records"]),
        f"{branch_id} reloaded score source records",
    )
    _require_equal(
        score.query_ids_sha256,
        identity["query_ids_sha256"],
        f"{branch_id} reloaded score query IDs",
    )
    _require_equal(
        score.gallery_ids_sha256,
        identity["gallery_ids_sha256"],
        f"{branch_id} reloaded score gallery IDs",
    )
    _require_equal(
        score.verified.payload_sha256,
        identity["score_payload_sha256"],
        f"{branch_id} reloaded score payload",
    )
    _require_equal(
        score.verified.envelope_sha256,
        identity["score_envelope_sha256"],
        f"{branch_id} reloaded score envelope",
    )
    alignment = common_alignment_payload(score)
    _require_equal(
        sha256_json(alignment),
        identity["alignment_sha256"],
        f"{branch_id} reloaded score alignment",
    )
    _require_equal(
        alignment,
        _clone_json(identity["alignment"]),
        f"{branch_id} reloaded score alignment body",
    )
    if branch_id == BRAINRW_BRANCH_ID:
        semantic_environment = _mapping(
            metadata.get("training_semantic_environment"),
            "BrainRW reloaded training semantic environment",
        )
        _require_equal(
            metadata.get("evaluation_semantic_environment"),
            semantic_environment,
            "BrainRW reloaded evaluation semantic environment",
        )
        _require_equal(
            semantic_environment,
            _clone_json(identity["semantic_environment"]),
            "BrainRW reloaded semantic environment",
        )


@dataclass(frozen=True)
class _CompletionBoundComponentRunProof(ValidatedComponentRunProof):
    _project_root: Path
    _branch_id: str
    _array_index: int
    _row_snapshot: bytes = field(repr=False)
    _completion_sha256: str
    _completion_document: bytes = field(repr=False)
    _completion_output_hashes: Mapping[str, str] = field(repr=False)
    _manifest_file_sha256: str
    _identity: Mapping[str, object] = field(repr=False)
    _proof_sha256: str
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _ISSUER_TOKEN:
            raise TypeError(
                "component run proofs can only be issued by the locked factory"
            )

    def revalidate(self) -> None:
        spec = _BRANCH_SPECS[self._branch_id]
        map_path = (
            self._project_root / _JOB_MAP_ROOT / str(spec["job_map_name"])
        )
        payload = load_job_map(
            map_path,
            expected_sha256=str(spec["job_map_sha256"]),
        )
        rows = _validate_locked_map(payload, branch_id=self._branch_id)
        row = rows[self._array_index]
        if canonical_json_bytes(row) != self._row_snapshot:
            raise ValueError("sealed component job-map row changed")
        completion = load_job_completion(payload, row)
        if completion is None:
            raise ValueError("component row has no current completion")
        completion.revalidate()
        if (
            completion.sha256 != self._completion_sha256
            or canonical_json_bytes(_clone_json(completion.document))
            != self._completion_document
            or dict(completion.output_hashes)
            != dict(self._completion_output_hashes)
        ):
            raise ValueError("current completion differs from issued proof")
        recipe = SemanticConfig.from_path(
            self._project_root
            / _CONFIG_ROOT
            / str(spec["config_name"])
        )
        if recipe.sha256 != spec["recipe_sha256"]:
            raise ValueError("component recipe config changed")
        config_path = (
            self._project_root
            / _CONFIG_ROOT
            / str(spec["config_name"])
        )
        _sha256_file(config_path)
        manifest_path = (
            self._project_root
            / _PROTOCOL_ROOT
            / f"sub-{int(row['subject']):02d}_protocol.json"
        )
        _require_equal(
            _sha256_file(manifest_path),
            self._manifest_file_sha256,
            f"{self._branch_id} protocol manifest file",
        )
        identity = _mapping(self._identity, "component proof identity")
        outputs = _mapping(
            identity["completion_output_hashes"],
            "component completion output hashes",
        )
        output_dir = Path(str(row["completion_path"])).parent
        checkpoint_name = (
            "checkpoint_epoch060.pt"
            if self._branch_id == INTERNVIT_BRANCH_ID
            else "checkpoint.pt"
        )
        _require_equal(
            _sha256_file(output_dir / checkpoint_name),
            outputs["final_checkpoint_sha256"],
            f"{self._branch_id} terminal checkpoint",
        )
        _require_equal(
            _sha256_file(output_dir / "run_manifest.json"),
            outputs["run_manifest_sha256"],
            f"{self._branch_id} run manifest",
        )
        if self._branch_id == INTERNVIT_BRANCH_ID:
            _require_equal(
                _sha256_file(output_dir / "baseline_parity.json"),
                outputs["parity_sha256"],
                "InternViT parity report",
            )
        score_directory = output_dir / str(spec["score_name"])
        score = _load_score_directory(score_directory)
        _require_equal(
            score.directory,
            score_directory,
            f"{self._branch_id} score directory",
        )
        _validate_reloaded_score(
            branch_id=self._branch_id,
            identity=identity,
            score=score,
        )
        _require_equal(
            _component_schedule(self._branch_id, recipe),
            _clone_json(identity["schedule"]),
            f"{self._branch_id} component schedule",
        )
        if self._proof_sha256 != sha256_json(identity):
            raise ValueError("component run proof SHA-256 mismatch")

    def identity_payload(self) -> dict[str, object]:
        result = _clone_json(self._identity)
        if not isinstance(result, dict):
            raise AssertionError("component proof identity is not an object")
        return result

    @property
    def proof_sha256(self) -> str:
        return self._proof_sha256


@dataclass(frozen=True)
class _IssuedComponent:
    branch_id: str
    subject: int
    seed: int
    score: ScoreArtifact
    proof: _CompletionBoundComponentRunProof


def _issue_component(
    *,
    branch_id: str,
    project_root: Path,
    payload: Mapping[str, object],
    recipe: SemanticConfig,
    row: Mapping[str, object],
) -> _IssuedComponent:
    completion = load_job_completion(payload, row)
    if completion is None:
        raise ValueError(
            f"{branch_id} row {row['array_index']} has no current completion"
        )
    completion.revalidate()
    validator = (
        validate_training_command_proof
        if branch_id == INTERNVIT_BRANCH_ID
        else validate_brainrw_command_proof
    )
    command_proof = validator(row["argv"], expected_mode="full")
    score = _validate_command_crossbindings(
        branch_id=branch_id,
        command_proof=command_proof,
        completion=completion,
        project_root=project_root,
        recipe=recipe,
        row=row,
    )
    identity = _identity_payload(
        branch_id=branch_id,
        command_proof=command_proof,
        completion=completion,
        recipe=recipe,
        row=row,
        score=score,
    )
    frozen_identity = _freeze_json(identity)
    if not isinstance(frozen_identity, Mapping):
        raise AssertionError("frozen component identity is not a mapping")
    frozen_outputs = MappingProxyType(dict(completion.output_hashes))
    proof = _CompletionBoundComponentRunProof(
        _project_root=project_root,
        _branch_id=branch_id,
        _array_index=int(row["array_index"]),
        _row_snapshot=canonical_json_bytes(row),
        _completion_sha256=completion.sha256,
        _completion_document=canonical_json_bytes(
            _clone_json(completion.document)
        ),
        _completion_output_hashes=frozen_outputs,
        _manifest_file_sha256=_sha256_file(
            project_root
            / _PROTOCOL_ROOT
            / f"sub-{int(row['subject']):02d}_protocol.json"
        ),
        _identity=frozen_identity,
        _proof_sha256=sha256_json(identity),
        _token=_ISSUER_TOKEN,
    )
    return _IssuedComponent(
        branch_id=branch_id,
        subject=int(row["subject"]),
        seed=int(row["seed"]),
        score=score,
        proof=proof,
    )


def _issue_locked_components(
    project_root: Path,
    semantic_config: SemanticConfig,
) -> tuple[_IssuedComponent, ...]:
    root, _ = _require_locked_semantic_config(project_root, semantic_config)
    issued: list[_IssuedComponent] = []
    for branch_id in (INTERNVIT_BRANCH_ID, BRAINRW_BRANCH_ID):
        spec = _BRANCH_SPECS[branch_id]
        recipe = SemanticConfig.from_path(
            root / _CONFIG_ROOT / str(spec["config_name"])
        )
        if recipe.sha256 != spec["recipe_sha256"]:
            raise ValueError(f"{branch_id} recipe config differs from locked v1")
        map_path = root / _JOB_MAP_ROOT / str(spec["job_map_name"])
        payload = load_job_map(
            map_path,
            expected_sha256=str(spec["job_map_sha256"]),
        )
        rows = _validate_locked_map(payload, branch_id=branch_id)
        issued.extend(
            _issue_component(
                branch_id=branch_id,
                project_root=root,
                payload=payload,
                recipe=recipe,
                row=row,
            )
            for row in rows
        )
    return tuple(issued)


def load_stage1_component_proofs(
    project_root: Path,
    semantic_config: SemanticConfig,
) -> tuple[ValidatedComponentRunProof, ...]:
    """Return the twelve locked proofs in branch-major pilot-grid order."""

    _require_locked_semantic_config(project_root, semantic_config)
    issued = _issue_locked_components(project_root, semantic_config)
    proofs = tuple(value.proof for value in issued)
    if len(proofs) != 12:
        raise ValueError("locked Stage 1 proof factory did not issue twelve proofs")
    return proofs


def load_stage1_composition_cells(
    project_root: Path,
    semantic_config: SemanticConfig,
) -> tuple[Stage1CompositionCell, ...]:
    """Return six fully bound Stage 1 cells in ``PILOT_COORDINATES`` order."""

    _require_locked_semantic_config(project_root, semantic_config)
    issued = _issue_locked_components(project_root, semantic_config)
    by_key = {
        (value.branch_id, value.subject, value.seed): value
        for value in issued
    }
    expected_keys = {
        (branch_id, subject, seed)
        for branch_id in (INTERNVIT_BRANCH_ID, BRAINRW_BRANCH_ID)
        for subject, seed in PILOT_COORDINATES
    }
    if len(by_key) != 12 or set(by_key) != expected_keys:
        raise ValueError("locked Stage 1 component proof grid is incomplete")
    cells = []
    for subject, seed in PILOT_COORDINATES:
        internvit = by_key[(INTERNVIT_BRANCH_ID, subject, seed)]
        brainrw = by_key[(BRAINRW_BRANCH_ID, subject, seed)]
        cells.append(
            Stage1CompositionCell(
                subject=subject,
                seed=seed,
                internvit=Stage1ComponentBinding(
                    branch_id=INTERNVIT_BRANCH_ID,
                    score=internvit.score,
                    run_proof=internvit.proof,
                ),
                brainrw=Stage1ComponentBinding(
                    branch_id=BRAINRW_BRANCH_ID,
                    score=brainrw.score,
                    run_proof=brainrw.proof,
                ),
            )
        )
    return tuple(cells)


__all__ = [
    "load_stage1_component_proofs",
    "load_stage1_composition_cells",
]
