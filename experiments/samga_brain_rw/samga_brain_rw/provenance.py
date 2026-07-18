"""Fail-closed, train-only provenance manifest construction.

The builder never opens a regular experiment input by raw path.  Callers must
first obtain a :class:`~samga_brain_rw.access.VerifiedArtifact` for every
regular file.  The builder re-verifies those descriptors, checks that every
capability is bound to its explicit path/type/scope, and then reads through
``VerifiedArtifact.open_verified()`` only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from .access import TypedArtifact, VerifiedArtifact, verify_typed_artifacts
from .hashing import canonical_json_bytes, sha256_json


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SUBJECT_TEST_RE = re.compile(r"^sub-\d{2}_test\.json$", re.IGNORECASE)
_FORMAL_TEST_RECORD_SHA256 = (
    "02d7e33b3fe8e5a571f8db232ca5fa86abb0c16981876ec84feae7ba64636f1a"
)
_MAX_JSON_BYTES = 64 * 1024 * 1024
_SUBJECTS = tuple(range(1, 11))


def _require_sha256(value: object, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _require_revision(value: object, context: str) -> str:
    if not isinstance(value, str) or _REVISION_RE.fullmatch(value) is None:
        raise ValueError(f"{context} must be a lowercase 40-character revision")
    return value


PACKAGE_VERSION_ALLOWLIST = (
    "numpy",
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "safetensors",
    "scipy",
    "scikit-learn",
)
ENVIRONMENT_VARIABLE_ALLOWLIST = (
    "HF_DATASETS_OFFLINE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
)


@dataclass(frozen=True)
class SourceFileOracle:
    """Pinned identity for one subject's source manifest and ``train.pt``."""

    subject_id: int
    manifest_sha256: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.subject_id) is not int or self.subject_id not in _SUBJECTS:
            raise ValueError("source oracle subject_id must be 1 through 10")
        _require_sha256(self.manifest_sha256, "manifest_sha256")
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ValueError("source oracle byte_count must be positive")
        _require_sha256(self.sha256, "source train.pt sha256")


@dataclass(frozen=True)
class SourceTrainPt:
    """Canonical output descriptor for one train-only EEG source."""

    subject_id: int
    manifest_path: Path
    manifest_sha256: str
    source_path: Path
    byte_count: int
    sha256: str

    def to_payload(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "sha256": self.sha256,
            "source_path": str(self.source_path),
            "subject_id": self.subject_id,
        }


@dataclass(frozen=True)
class ProvenanceOracles:
    """The complete sealed provenance contract.

    Tests may replace values with small synthetic fixture identities.  The
    public default remains the only production oracle set.
    """

    upstream_revision: str
    cache_generator_revision: str
    model_revision: str
    protocol_config_sha256: str
    internvit_semantic_config_sha256: str
    brainrw_semantic_config_sha256: str
    canonical_records_sha256: str
    canonical_train_manifest_sha256: str
    split_assignment_file_sha256: str
    split_assignment_payload_sha256: str
    manifest_summary_file_sha256: str
    record_count: int
    concept_count: int
    stimuli_per_concept: int
    source_files: tuple[SourceFileOracle, ...]
    internvit_repository: str
    internvit_config_sha256: str
    internvit_preprocessor_sha256: str
    internvit_modeling_sha256: str
    internvit_weight_sha256: tuple[str, str, str]
    selected_cache_declared_path: str
    selected_cache_sha256: str
    selected_cache_size: int
    selected_cache_shape: tuple[int, int, int]
    merged_cache_sha256: str
    merged_cache_shape: tuple[int, int, int]
    clip_model_id: str
    clip_config_sha256: str
    clip_weights_sha256: str
    clip_cache_sha256: str
    clip_cache_size: int
    clip_cache_shape: tuple[int, int, int]
    data_root: str
    clip_model_path: str

    def __post_init__(self) -> None:
        for name in (
            "upstream_revision",
            "cache_generator_revision",
            "model_revision",
        ):
            _require_revision(getattr(self, name), name)
        for name in (
            "protocol_config_sha256",
            "internvit_semantic_config_sha256",
            "brainrw_semantic_config_sha256",
            "canonical_records_sha256",
            "canonical_train_manifest_sha256",
            "split_assignment_file_sha256",
            "split_assignment_payload_sha256",
            "manifest_summary_file_sha256",
            "internvit_config_sha256",
            "internvit_preprocessor_sha256",
            "internvit_modeling_sha256",
            "selected_cache_sha256",
            "merged_cache_sha256",
            "clip_config_sha256",
            "clip_weights_sha256",
            "clip_cache_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if len(self.internvit_weight_sha256) != 3:
            raise ValueError("exactly three InternViT weight hashes are required")
        for digest in self.internvit_weight_sha256:
            _require_sha256(digest, "InternViT weight sha256")
        if tuple(item.subject_id for item in self.source_files) != _SUBJECTS:
            raise ValueError("source oracles must contain subjects 1 through 10")
        for name in ("record_count", "concept_count", "stimuli_per_concept"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.record_count != self.concept_count * self.stimuli_per_concept:
            raise ValueError("record_count must equal concepts times stimuli")
        for name in (
            "selected_cache_shape",
            "merged_cache_shape",
            "clip_cache_shape",
        ):
            shape = getattr(self, name)
            if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
                raise ValueError(f"{name} must be a positive rank-3 shape")
        for name in ("selected_cache_size", "clip_cache_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "internvit_repository",
            "selected_cache_declared_path",
            "clip_model_id",
            "data_root",
            "clip_model_path",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")


DEFAULT_ORACLES = ProvenanceOracles(
    upstream_revision="1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1",
    cache_generator_revision="a97b97a110c0fea7d4adafd5abce477c6cce525c",
    model_revision="9d1a4344077479c93d42584b6941c64d795d508d",
    protocol_config_sha256=(
        "0a9bb1dc750145ec94c35aaaddf5a834d303be3e6f69c9740237d9b967fd48bd"
    ),
    internvit_semantic_config_sha256=(
        "db3a696a31ceba0699c4039ca73130f75edd7d2ad69ce3e55c9ab7a5ecfc27de"
    ),
    brainrw_semantic_config_sha256=(
        "7224907692b8516a2b07c5b6dcc242288a9a0738259c71abae8674d9cb99e53d"
    ),
    canonical_records_sha256=(
        "f59500f36e273f66fce5c2019670b076d75d538feccf296c7d7ed75f19ae3fac"
    ),
    canonical_train_manifest_sha256=(
        "42fd7316314eb02d69ee2234d4d8430afcfcc2a5f6834e9c7be64f38eccdbc85"
    ),
    split_assignment_file_sha256=(
        "1d5ad2344797b359a3aeb04f1c298a7785b1492f2eee44e1d1a178e929ad70dc"
    ),
    split_assignment_payload_sha256=(
        "4463e408af8644eed4c73a4d82832d402ba0b4f70b338f2e797216fd3698d912"
    ),
    manifest_summary_file_sha256=(
        "bbd84cd87dda3ac5f02270a03923857e1e79c13a6acaa4c0a4d4556a1c413dce"
    ),
    record_count=16_540,
    concept_count=1_654,
    stimuli_per_concept=10,
    source_files=(
        SourceFileOracle(
            1,
            "42fd7316314eb02d69ee2234d4d8430afcfcc2a5f6834e9c7be64f38eccdbc85",
            2_106_803_096,
            "815931d6e9eadcda5b1edd054248f81c1e9a42dd4abafe5eb075a80269090f20",
        ),
        SourceFileOracle(
            2,
            "1d6275829da9f423c090d48350dfe106ac27759225265b9a3c796ddb4f77d0a0",
            2_106_802_667,
            "b3e797dc48e8585e36b7a71a6fb40f837061e62fc24c992cab84431fdb110d21",
        ),
        SourceFileOracle(
            3,
            "123ba9dfdd983173fe6b5f6a739c515ca2b12ab101898549756a6d9a8462086e",
            2_106_802_667,
            "94acbc554a5b179c19e280b2f8025704c91294be610f425d85c58c8f33f83776",
        ),
        SourceFileOracle(
            4,
            "eb133c98c761de61bb87154dd140df2f82047512fbd8e170a50e7cfaf005e7e5",
            2_106_802_667,
            "d2a24081c559e77c88adbe152018f48476f5e5ae4995fe7496aaab72660a92f8",
        ),
        SourceFileOracle(
            5,
            "f278c3a6efafeffc278b871ae111792fbb0cf41ee05cd11e6e24d3497afd7b6b",
            2_106_802_667,
            "ded2b3f518347e9aa0a5dd9efe3744a35a7f19edcd743564c09d162d38b8890d",
        ),
        SourceFileOracle(
            6,
            "a88bdf485d0d05548c45ffda0b9fdbd9aad69207bcc88b258ea860da0d7244e8",
            2_106_802_667,
            "11768a5fa84f76c4b8067ab950df40bf906e71c75ff656cb8a1dbfbdef2c07b3",
        ),
        SourceFileOracle(
            7,
            "12c6629989cf6b0fdf0aff963c0f690f21a2e46978b40aed54a39e3230d8d52b",
            2_106_802_667,
            "b858327f80d1b0fa4fffc1c9da26d72786ec9ee4833ce462feae4b136ce68a9a",
        ),
        SourceFileOracle(
            8,
            "703f9e305822da747c4fa5ee61c277578e5e7d3da42947bf2b17742909e3425d",
            2_106_803_096,
            "c5997c48df5e1ef976066f63f6d13dc8ed9babaabe8d75a8f2d8592770c018b3",
        ),
        SourceFileOracle(
            9,
            "6d30eca14797961805d3d113de2cbabbc448f1f3f83abb48f51c3565e440377b",
            2_106_802_667,
            "aec05e4dfeb822a30cba6b5955927a8bd1d9a4dee8d007a65b69bd11d36b19c1",
        ),
        SourceFileOracle(
            10,
            "abde70e302375e9ca3d94c5d2ce593e4be699fe817e17bdfc255112d8523483e",
            2_106_802_667,
            "61cb24fc9269d7760af9d12e2f32439c65a7aea5ae59f8c4efc6109a437fdeee",
        ),
    ),
    internvit_repository="OpenGVLab/InternViT-6B-448px-V2_5",
    internvit_config_sha256=(
        "4fc4a1187b20575c0da8d27df2ad17f5ad6e8ac1c8b2af707bc8b263bd40c0a2"
    ),
    internvit_preprocessor_sha256=(
        "0658115064c561026539aeeead9ed3b1a8e0cc90967df8c142849199f955d2b4"
    ),
    internvit_modeling_sha256=(
        "56220ba82cb511d51d5f2fa71eebd728b330fbccad9dfb128088a8fcc8f7d260"
    ),
    internvit_weight_sha256=(
        "9818659d13d932da8bc0c3b8ee15f5b5d68d8c94d66eb525be566066630111da",
        "4f0c10e72d6f6513f421baa6ec843d5508657435059c1d18b6b5fd7789f9d5b7",
        "d21c4fe0bc4af1425cfae1a59a8f5fbb00fde9d8e2888325a60913ac61b0494d",
    ),
    selected_cache_declared_path=(
        "artifacts/samga_reproduction/features/internvit_v2_5_multi_variant/"
        "variants/train_idx0_patch_mean/features.npy"
    ),
    selected_cache_sha256=(
        "539c7b62ae41c8112e22b3ddc3a6566d997465a10c36d16c8f2378855ba94c71"
    ),
    selected_cache_size=529_280_128,
    selected_cache_shape=(16_540, 5, 3_200),
    merged_cache_sha256=(
        "e5b92033c3b0fd19d71ca825844568883a7fea85c5e97fd6b21db445ff93e1dc"
    ),
    merged_cache_shape=(16_540, 10, 3_200),
    clip_model_id="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    clip_config_sha256=(
        "1284cbff35169abb23a1c5525a8b0f543c7bd191d4b9aed63880c1571bc4191c"
    ),
    clip_weights_sha256=(
        "74813fbcdc750f235c9784c367ca1394d2a5c25eb0aac92761752ac239db7cff"
    ),
    clip_cache_sha256=(
        "a31c1871082e1f052da3d055702455b464ea2345890eee33e447e09328c45ebb"
    ),
    clip_cache_size=127_027_328,
    clip_cache_shape=(16_540, 5, 768),
    data_root=(
        "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/"
        "EEG_Recon-RL/datasets/things_eeg_data"
    ),
    clip_model_path=(
        "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/"
        "CLIP-ViT-B-32-laion2B-s34B-b79K"
    ),
)


@dataclass(frozen=True)
class EnvironmentSnapshot:
    python_version: str
    python_executable: str
    sys_prefix: str
    platform: str
    machine: str
    hostname: str
    package_versions: Mapping[str, str | None]
    selected_environment: Mapping[str, str | None]

    def __post_init__(self) -> None:
        for name in (
            "python_version",
            "python_executable",
            "sys_prefix",
            "platform",
            "machine",
            "hostname",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        packages = dict(self.package_versions)
        selected = dict(self.selected_environment)
        if set(packages) != set(PACKAGE_VERSION_ALLOWLIST):
            raise ValueError("package version keys must equal the strict allowlist")
        if set(selected) != set(ENVIRONMENT_VARIABLE_ALLOWLIST):
            raise ValueError(
                "selected environment keys must equal the strict allowlist"
            )
        for mapping_name, mapping in (
            ("package_versions", packages),
            ("selected_environment", selected),
        ):
            if any(
                value is not None and not isinstance(value, str)
                for value in mapping.values()
            ):
                raise ValueError(f"{mapping_name} values must be strings or null")
        object.__setattr__(self, "package_versions", MappingProxyType(packages))
        object.__setattr__(self, "selected_environment", MappingProxyType(selected))

    def to_payload(self) -> dict[str, object]:
        return {
            "hostname": self.hostname,
            "machine": self.machine,
            "package_versions": {
                key: self.package_versions[key]
                for key in PACKAGE_VERSION_ALLOWLIST
            },
            "platform": self.platform,
            "python_executable": self.python_executable,
            "python_version": self.python_version,
            "selected_environment": {
                key: self.selected_environment[key]
                for key in ENVIRONMENT_VARIABLE_ALLOWLIST
            },
            "sys_prefix": self.sys_prefix,
        }


@dataclass(frozen=True)
class ProvenanceInputs:
    repository_root: Path
    protocol_path: Path
    internvit_config_path: Path
    brainrw_config_path: Path
    source_manifest_dir: Path
    protocol_manifest_dir: Path
    feature_directory: Path
    variant_directory: Path
    canonical_cache: Path
    clip_train_cache: Path
    data_root: Path
    model_path: Path
    clip_model_path: Path
    upstream_root: Path
    experiment_revision: str
    upstream_revision: str
    cache_generator_revision: str
    verified_artifacts: Mapping[str, VerifiedArtifact]
    environment: EnvironmentSnapshot
    oracles: ProvenanceOracles = DEFAULT_ORACLES

    def __post_init__(self) -> None:
        for name in (
            "repository_root",
            "protocol_path",
            "internvit_config_path",
            "brainrw_config_path",
            "source_manifest_dir",
            "protocol_manifest_dir",
            "feature_directory",
            "variant_directory",
            "canonical_cache",
            "clip_train_cache",
            "data_root",
            "model_path",
            "clip_model_path",
            "upstream_root",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "experiment_revision",
            "upstream_revision",
            "cache_generator_revision",
        ):
            _require_revision(getattr(self, name), name)
        if not isinstance(self.environment, EnvironmentSnapshot):
            raise TypeError("environment must be an EnvironmentSnapshot")
        if not isinstance(self.oracles, ProvenanceOracles):
            raise TypeError("oracles must be ProvenanceOracles")
        capabilities = dict(self.verified_artifacts)
        if any(
            not isinstance(key, str) or not isinstance(value, VerifiedArtifact)
            for key, value in capabilities.items()
        ):
            raise TypeError(
                "verified_artifacts must map string keys to VerifiedArtifact values"
            )
        object.__setattr__(
            self,
            "verified_artifacts",
            MappingProxyType(capabilities),
        )


def _build_capability_types() -> dict[str, str]:
    result = {
        "protocol": "samga_brain_rw.protocol_config",
        "internvit_config": "samga_brain_rw.semantic_config",
        "brainrw_config": "samga_brain_rw.semantic_config",
        "protocol.split_assignment": "samga_brain_rw.split_assignment",
        "protocol.manifest_summary": "samga_brain_rw.manifest_summary",
    }
    for subject in _SUBJECTS:
        result[f"source_manifest.sub-{subject:02d}"] = (
            "samga_brain_rw.source_manifest"
        )
    for subject in _SUBJECTS:
        result[f"protocol_manifest.sub-{subject:02d}"] = (
            "samga_brain_rw.role_payload"
        )
    for subject in _SUBJECTS:
        result[f"source_train_pt.sub-{subject:02d}"] = (
            "samga_brain_rw.source_train_pt"
        )
    result.update(
        {
            "internvit.config": "samga_brain_rw.model_config",
            "internvit.preprocessor": "samga_brain_rw.model_preprocessor",
            "internvit.modeling": "samga_brain_rw.model_source",
            "internvit.weight.1": "samga_brain_rw.model_weights",
            "internvit.weight.2": "samga_brain_rw.model_weights",
            "internvit.weight.3": "samga_brain_rw.model_weights",
            "clip.config": "samga_brain_rw.model_config",
            "clip.weights": "samga_brain_rw.model_weights",
            "cache.internvit_selected": "samga_brain_rw.train_cache",
            "cache.internvit_merged": "samga_brain_rw.train_cache",
            "cache.clip_train": "samga_brain_rw.train_cache",
            "cache.internvit_selected_metadata": (
                "samga_brain_rw.train_cache_metadata"
            ),
            "cache.internvit_merged_metadata": (
                "samga_brain_rw.train_cache_metadata"
            ),
            "cache.clip_train_metadata": (
                "samga_brain_rw.train_cache_metadata"
            ),
        }
    )
    return result


CAPABILITY_PAYLOAD_TYPES = MappingProxyType(_build_capability_types())


def expected_capability_paths(inputs: ProvenanceInputs) -> dict[str, Path]:
    """Return every exact regular-file path; no directory is scanned."""

    paths: dict[str, Path] = {
        "protocol": inputs.protocol_path,
        "internvit_config": inputs.internvit_config_path,
        "brainrw_config": inputs.brainrw_config_path,
        "protocol.split_assignment": (
            inputs.protocol_manifest_dir / "split_assignment.json"
        ),
        "protocol.manifest_summary": (
            inputs.protocol_manifest_dir / "manifest_summary.json"
        ),
    }
    for subject in _SUBJECTS:
        paths[f"source_manifest.sub-{subject:02d}"] = (
            inputs.source_manifest_dir / f"sub-{subject:02d}_train.json"
        )
    for subject in _SUBJECTS:
        paths[f"protocol_manifest.sub-{subject:02d}"] = (
            inputs.protocol_manifest_dir / f"sub-{subject:02d}_protocol.json"
        )
    for subject in _SUBJECTS:
        paths[f"source_train_pt.sub-{subject:02d}"] = (
            inputs.data_root
            / "Preprocessed_data_250Hz_whiten"
            / f"sub-{subject:02d}"
            / "train.pt"
        )
    paths.update(
        {
            "internvit.config": inputs.model_path / "config.json",
            "internvit.preprocessor": (
                inputs.model_path / "preprocessor_config.json"
            ),
            "internvit.modeling": inputs.model_path / "modeling_intern_vit.py",
            "internvit.weight.1": (
                inputs.model_path / "model-00001-of-00003.safetensors"
            ),
            "internvit.weight.2": (
                inputs.model_path / "model-00002-of-00003.safetensors"
            ),
            "internvit.weight.3": (
                inputs.model_path / "model-00003-of-00003.safetensors"
            ),
            "clip.config": inputs.clip_model_path / "config.json",
            "clip.weights": inputs.clip_model_path / "model.safetensors",
            "cache.internvit_selected": inputs.canonical_cache,
            "cache.internvit_merged": inputs.feature_directory / "patch_mean.npy",
            "cache.clip_train": inputs.clip_train_cache,
            "cache.internvit_selected_metadata": (
                inputs.variant_directory / "metadata.json"
            ),
            "cache.internvit_merged_metadata": (
                inputs.feature_directory / "metadata.json"
            ),
            "cache.clip_train_metadata": Path(
                f"{inputs.clip_train_cache}.meta.json"
            ),
        }
    )
    if tuple(paths) != tuple(CAPABILITY_PAYLOAD_TYPES):
        raise AssertionError("capability path/type registries diverged")
    return paths


def preflight_provenance_inputs(
    inputs: ProvenanceInputs,
) -> dict[str, Path]:
    """Validate paths and repository identities before descriptor opens."""

    if not isinstance(inputs, ProvenanceInputs):
        raise TypeError("inputs must be ProvenanceInputs")
    paths = expected_capability_paths(inputs)
    _preflight_input_paths(inputs, paths)
    if _normalized(inputs.data_root) != _normalized(
        Path(inputs.oracles.data_root)
    ):
        raise ValueError("input does not match the pinned data root")
    if _normalized(inputs.clip_model_path) != _normalized(
        Path(inputs.oracles.clip_model_path)
    ):
        raise ValueError("input does not match the pinned CLIP model path")
    experiment_head = _read_git_head(
        inputs.repository_root, "experiment repository"
    )
    upstream_head = _read_git_head(inputs.upstream_root, "upstream repository")
    if experiment_head != inputs.experiment_revision:
        raise ValueError("experiment repository HEAD mismatch")
    if upstream_head != inputs.upstream_revision:
        raise ValueError("upstream repository HEAD mismatch")
    if inputs.upstream_revision != inputs.oracles.upstream_revision:
        raise ValueError("upstream revision mismatch")
    if (
        inputs.cache_generator_revision
        != inputs.oracles.cache_generator_revision
    ):
        raise ValueError("cache-generator revision mismatch")
    return paths


def build_provenance_manifest(
    inputs: ProvenanceInputs,
) -> dict[str, object]:
    """Validate and return the canonical train-only provenance payload."""

    if not isinstance(inputs, ProvenanceInputs):
        raise TypeError("inputs must be ProvenanceInputs")
    paths = preflight_provenance_inputs(inputs)
    oracles = inputs.oracles
    capabilities = _bind_and_reverify_capabilities(inputs, paths)

    protocol = _read_json(capabilities["protocol"], "protocol config")
    internvit = _read_json(
        capabilities["internvit_config"], "InternViT semantic config"
    )
    brainrw = _read_json(
        capabilities["brainrw_config"], "brain-rw semantic config"
    )
    _validate_protocol(protocol, oracles)
    _validate_semantic_configs(internvit, brainrw, inputs, oracles)
    protocol_registry = _validate_protocol_registry(
        capabilities, inputs, oracles
    )

    subjects, source_records = _validate_sources(
        inputs,
        capabilities,
        oracles,
    )
    _validate_protocol_capabilities(capabilities)
    _validate_model_files(capabilities, oracles)
    cache_payload = _validate_caches(capabilities, inputs, internvit, oracles)

    model_config = _read_json(
        capabilities["internvit.config"], "InternViT model config"
    )
    model_config_transformers = model_config.get("transformers_version")
    if model_config_transformers is not None and not isinstance(
        model_config_transformers, str
    ):
        raise ValueError("model config transformers_version must be a string")

    weight_names = (
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    )
    manifest = {
        "schema_version": 1,
        "payload_type": "samga_brain_rw.provenance_manifest",
        "scope": "train",
        "passed": True,
        "protocol": {
            "brainrw_config": {
                "path": str(_normalized(inputs.brainrw_config_path)),
                "sha256": oracles.brainrw_semantic_config_sha256,
            },
            "internvit_config": {
                "path": str(_normalized(inputs.internvit_config_path)),
                "sha256": oracles.internvit_semantic_config_sha256,
            },
            "path": str(_normalized(inputs.protocol_path)),
            "protocol_manifest_dir": str(
                _normalized(inputs.protocol_manifest_dir)
            ),
            "registry": protocol_registry,
            "sha256": oracles.protocol_config_sha256,
        },
        "repositories": {
            "cache_generator_revision": inputs.cache_generator_revision,
            "experiment_revision": inputs.experiment_revision,
            "upstream_revision": inputs.upstream_revision,
        },
        "data": {
            "canonical_train_manifest_sha256": (
                oracles.canonical_train_manifest_sha256
            ),
            "concept_count": oracles.concept_count,
            "record_count": oracles.record_count,
            "records_sha256": hashlib.sha256(
                canonical_json_bytes(source_records)
            ).hexdigest(),
            "root": str(_normalized(inputs.data_root)),
            "stimuli_per_concept": oracles.stimuli_per_concept,
            "subjects": [subject.to_payload() for subject in subjects],
        },
        "models": {
            "clip": {
                "config": {
                    "path": str(_normalized(paths["clip.config"])),
                    "sha256": oracles.clip_config_sha256,
                },
                "model_id": oracles.clip_model_id,
                "path": str(_normalized(inputs.clip_model_path)),
                "weights": {
                    "path": str(_normalized(paths["clip.weights"])),
                    "sha256": oracles.clip_weights_sha256,
                },
            },
            "internvit": {
                "config": {
                    "path": str(_normalized(paths["internvit.config"])),
                    "sha256": oracles.internvit_config_sha256,
                    "transformers_version": model_config_transformers,
                },
                "modeling_source": {
                    "path": str(_normalized(paths["internvit.modeling"])),
                    "sha256": oracles.internvit_modeling_sha256,
                },
                "path": str(_normalized(inputs.model_path)),
                "preprocessor": {
                    "path": str(_normalized(paths["internvit.preprocessor"])),
                    "sha256": oracles.internvit_preprocessor_sha256,
                },
                "repository": oracles.internvit_repository,
                "revision": oracles.model_revision,
                "weights": [
                    {
                        "filename": filename,
                        "sha256": digest,
                    }
                    for filename, digest in zip(
                        weight_names,
                        oracles.internvit_weight_sha256,
                        strict=True,
                    )
                ],
            },
        },
        "caches": cache_payload,
        "extraction_semantics": {
            "downstream_image_l2_normalization": True,
            "extractor_normalization": "none",
            "layer_route": "idx0",
            "logical_layers": [20, 24, 28, 32, 36],
            "pooling": "mean(hidden[:,1:,:], axis=1)",
            "source_axes": [0, 2, 4, 6, 8],
        },
        "environment": inputs.environment.to_payload(),
        "checks": {
            "cache_headers": True,
            "cache_metadata": True,
            "capabilities_reverified": True,
            "config_hashes": True,
            "environment_allowlist": True,
            "model_hashes": True,
            "paths_train_only": True,
            "repository_revisions": True,
            "source_counts": True,
            "source_hashes": True,
            "source_records": True,
            "train_protocol_roles": True,
        },
    }
    if not all(manifest["checks"].values()):  # pragma: no cover - constants
        raise AssertionError("provenance manifest cannot publish failed checks")
    return manifest


def _preflight_input_paths(
    inputs: ProvenanceInputs,
    regular_files: Mapping[str, Path],
) -> None:
    all_paths = (
        inputs.repository_root,
        inputs.protocol_path,
        inputs.internvit_config_path,
        inputs.brainrw_config_path,
        inputs.source_manifest_dir,
        inputs.protocol_manifest_dir,
        inputs.feature_directory,
        inputs.variant_directory,
        inputs.canonical_cache,
        inputs.clip_train_cache,
        inputs.data_root,
        inputs.model_path,
        inputs.clip_model_path,
        inputs.upstream_root,
        *regular_files.values(),
    )
    for path in all_paths:
        _reject_forbidden_path(path)
    expected_selected = inputs.variant_directory / "features.npy"
    if _normalized(inputs.canonical_cache) != _normalized(expected_selected):
        raise ValueError("canonical cache path must be variant_directory/features.npy")


def _reject_forbidden_path(path: Path) -> None:
    raw = os.fspath(path)
    if not isinstance(raw, str) or not raw:
        raise ValueError("input path must be non-empty text")
    if "\x00" in raw:
        raise ValueError("input path is forbidden: NUL byte")
    normalized = os.path.abspath(os.path.normpath(raw))
    lowered = normalized.lower()
    if _FORMAL_TEST_RECORD_SHA256 in lowered:
        raise ValueError("input path is forbidden: formal-test digest")
    path_value = Path(normalized)
    if _SUBJECT_TEST_RE.fullmatch(path_value.name):
        raise ValueError("input path is forbidden: subject test manifest")
    forbidden_components = {"test_images", "val-confirm", "formal-test"}
    if any(part.lower() in forbidden_components for part in path_value.parts):
        raise ValueError("input path is forbidden: sealed scope component")
    _reject_symlink_components(path_value)


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ValueError("input path cannot be inspected safely") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise ValueError("input path is forbidden: symlink component")


def _read_git_head(root: Path, context: str) -> str:
    """Read one explicit repository HEAD without scanning any worktree file."""

    normalized = _normalized(root)
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(normalized),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"{context} HEAD could not be verified") from exc
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise ValueError(f"{context} returned an invalid HEAD")
    head = lines[0].strip()
    _require_revision(head, f"{context} HEAD")
    return head


def _bind_and_reverify_capabilities(
    inputs: ProvenanceInputs,
    paths: Mapping[str, Path],
) -> dict[str, VerifiedArtifact]:
    provided = dict(inputs.verified_artifacts)
    missing = set(paths) - set(provided)
    extra = set(provided) - set(paths)
    if missing:
        raise ValueError(
            f"missing verified artifact capability: {sorted(missing)}"
        )
    if extra:
        raise ValueError(f"unknown verified artifact capability: {sorted(extra)}")
    descriptors: list[TypedArtifact] = []
    for key, expected_path in paths.items():
        capability = provided[key]
        descriptor = capability.artifact
        if _normalized(descriptor.payload_path) != _normalized(expected_path):
            raise ValueError(f"{key} capability path mismatch")
        expected_type = CAPABILITY_PAYLOAD_TYPES[key]
        if descriptor.payload_type != expected_type:
            raise ValueError(f"{key} capability payload type mismatch")
        expected_role = "train" if key.startswith("protocol_manifest.") else None
        if descriptor.role != expected_role:
            raise ValueError(f"{key} capability role mismatch")
        if capability.scope != "train":
            raise ValueError(f"{key} capability scope must be train")
        descriptors.append(descriptor)

    fresh = verify_typed_artifacts("train", descriptors)
    if len(fresh) != len(descriptors):
        raise ValueError("typed artifact verifier returned the wrong count")
    result: dict[str, VerifiedArtifact] = {}
    for key, new in zip(paths, fresh, strict=True):
        old = provided[key]
        if old.artifact != new.artifact or _capability_identity(old) != (
            _capability_identity(new)
        ):
            raise ValueError(f"{key} capability changed during re-verification")
        result[key] = new
    return result


def _capability_identity(
    value: VerifiedArtifact,
) -> tuple[int, int, int, int, int, str, str]:
    return (
        value.device,
        value.inode,
        value.size,
        value.mtime_ns,
        value.ctime_ns,
        value.payload_sha256,
        value.scope,
    )


def _validate_protocol(
    protocol: Mapping[str, object],
    oracles: ProvenanceOracles,
) -> None:
    if sha256_json(protocol) != oracles.protocol_config_sha256:
        raise ValueError("protocol semantic SHA-256 mismatch")
    if protocol.get("schema_version") != 1:
        raise ValueError("protocol schema_version mismatch")
    if protocol.get("expected_non_test_concepts") != oracles.concept_count:
        raise ValueError("protocol concept count mismatch")
    split_sizes = _mapping(protocol.get("split_sizes"), "protocol split_sizes")
    split_total = sum(
        _integer(value, "protocol split size")
        for value in split_sizes.values()
    )
    if split_total != oracles.concept_count:
        raise ValueError("protocol split concept counts mismatch")
    retrieval = _mapping(protocol.get("retrieval"), "protocol retrieval")
    expected_retrieval = {
        "method": "standard_independent_cosine",
        "similarity": "cosine",
        "assignment": "independent",
        "hungarian": False,
    }
    if dict(retrieval) != expected_retrieval:
        raise ValueError("protocol retrieval semantics mismatch")


def _validate_semantic_configs(
    internvit: Mapping[str, object],
    brainrw: Mapping[str, object],
    inputs: ProvenanceInputs,
    oracles: ProvenanceOracles,
) -> None:
    if sha256_json(internvit) != oracles.internvit_semantic_config_sha256:
        raise ValueError("InternViT semantic config SHA-256 mismatch")
    if sha256_json(brainrw) != oracles.brainrw_semantic_config_sha256:
        raise ValueError("brain-rw semantic config SHA-256 mismatch")
    upstream = _mapping(internvit.get("upstream"), "InternViT upstream")
    if _normalized(Path(_string(upstream.get("path"), "upstream path"))) != (
        _normalized(inputs.upstream_root)
    ):
        raise ValueError("upstream path mismatch")
    if upstream.get("git_commit") != oracles.upstream_revision:
        raise ValueError("InternViT upstream revision mismatch")
    model = _mapping(internvit.get("model"), "InternViT model")
    if model.get("repo") != oracles.internvit_repository:
        raise ValueError("InternViT repository mismatch")
    if model.get("revision") != oracles.model_revision:
        raise ValueError("InternViT model revision mismatch")
    if _normalized(Path(_string(model.get("path"), "model path"))) != _normalized(
        inputs.model_path
    ):
        raise ValueError("InternViT model path mismatch")
    if model.get("config_sha256") != oracles.internvit_config_sha256:
        raise ValueError("InternViT config digest mismatch")
    if model.get("preprocessor_sha256") != oracles.internvit_preprocessor_sha256:
        raise ValueError("InternViT preprocessor digest mismatch")
    weight_mapping = _mapping(model.get("weight_sha256"), "model weight hashes")
    expected_weight_mapping = {
        f"model-0000{index}-of-00003.safetensors": digest
        for index, digest in enumerate(oracles.internvit_weight_sha256, start=1)
    }
    if dict(weight_mapping) != expected_weight_mapping:
        raise ValueError("InternViT weight digest mapping mismatch")

    cache = _mapping(internvit.get("cache"), "InternViT cache")
    expected_cache_fields = {
        "path": oracles.selected_cache_declared_path,
        "sha256": oracles.selected_cache_sha256,
        "generator_git_revision": oracles.cache_generator_revision,
        "canonical_train_manifest_sha256": (
            oracles.canonical_train_manifest_sha256
        ),
        "shape": list(oracles.selected_cache_shape),
        "dtype": "float16",
        "layer_route": "idx0",
        "pooling": "patch_mean",
        "normalization": "none",
    }
    if dict(cache) != expected_cache_fields:
        raise ValueError("InternViT cache semantics mismatch")
    task = _mapping(internvit.get("task"), "InternViT task")
    semantic_expectations = {
        "layer_ids": [20, 24, 28, 32, 36],
        "image_dim": 3200,
        "prior_center": 28,
        "image_l2_normalization": True,
    }
    for key, expected in semantic_expectations.items():
        if task.get(key) != expected:
            raise ValueError(f"InternViT task {key} mismatch")

    clip = _mapping(brainrw.get("clip"), "brain-rw clip")
    if clip.get("model_id") != oracles.clip_model_id:
        raise ValueError("CLIP model ID mismatch")
    if _normalized(Path(_string(clip.get("path"), "CLIP path"))) != _normalized(
        inputs.clip_model_path
    ):
        raise ValueError("CLIP model path mismatch")
    if clip.get("config_sha256") != oracles.clip_config_sha256:
        raise ValueError("CLIP config digest mismatch")
    if clip.get("weights_sha256") != oracles.clip_weights_sha256:
        raise ValueError("CLIP weights digest mismatch")


def _validate_protocol_registry(
    capabilities: Mapping[str, VerifiedArtifact],
    inputs: ProvenanceInputs,
    oracles: ProvenanceOracles,
) -> dict[str, object]:
    split_capability = capabilities["protocol.split_assignment"]
    summary_capability = capabilities["protocol.manifest_summary"]
    if (
        split_capability.payload_sha256
        != oracles.split_assignment_file_sha256
    ):
        raise ValueError("split-assignment file SHA-256 mismatch")
    if (
        summary_capability.payload_sha256
        != oracles.manifest_summary_file_sha256
    ):
        raise ValueError("manifest-summary file SHA-256 mismatch")
    split = _read_json(split_capability, "split assignment")
    if sha256_json(split) != oracles.split_assignment_payload_sha256:
        raise ValueError("split-assignment semantic payload SHA-256 mismatch")
    if split.get("payload_type") != "samga_brain_rw.split_assignment":
        raise ValueError("split-assignment payload_type mismatch")
    if split.get("schema_version") != 1:
        raise ValueError("split-assignment schema_version mismatch")
    if split.get("protocol_config_sha256") != oracles.protocol_config_sha256:
        raise ValueError("split-assignment protocol binding mismatch")
    if split.get("records_sha256") != oracles.canonical_records_sha256:
        raise ValueError("split-assignment records binding mismatch")
    if split.get("record_count") != oracles.record_count:
        raise ValueError("split-assignment record count mismatch")

    summary = _read_json(summary_capability, "manifest summary")
    if summary.get("payload_type") != "samga_brain_rw.manifest_summary":
        raise ValueError("manifest-summary payload_type mismatch")
    if summary.get("schema_version") != 1:
        raise ValueError("manifest-summary schema_version mismatch")
    expected_summary_fields = {
        "protocol_config_sha256": oracles.protocol_config_sha256,
        "records_sha256": oracles.canonical_records_sha256,
        "record_count_per_subject": oracles.record_count,
        "split_assignment_file_sha256": (
            oracles.split_assignment_file_sha256
        ),
        "split_assignment_payload_sha256": (
            oracles.split_assignment_payload_sha256
        ),
        "subject_count": 10,
    }
    for key, expected in expected_summary_fields.items():
        if summary.get(key) != expected:
            raise ValueError(f"manifest-summary {key} mismatch")

    summary_subjects = summary.get("subjects")
    if not isinstance(summary_subjects, list) or len(summary_subjects) != 10:
        raise ValueError("manifest-summary subjects must contain all 10 subjects")
    expected_subject_keys = {
        "protocol_manifest",
        "protocol_manifest_sha256",
        "source_manifest_path",
        "source_manifest_sha256",
        "subject_id",
    }
    for index, (raw_subject, source_oracle) in enumerate(
        zip(summary_subjects, oracles.source_files, strict=True)
    ):
        context = f"manifest-summary subjects[{index}]"
        subject = _mapping(raw_subject, context)
        if set(subject) != expected_subject_keys:
            raise ValueError(f"{context} key mismatch")
        subject_id = source_oracle.subject_id
        if subject.get("subject_id") != subject_id:
            raise ValueError(f"{context} subject_id mismatch")
        label = f"sub-{subject_id:02d}"

        protocol_filename = _string(
            subject.get("protocol_manifest"),
            f"{context} protocol_manifest",
        )
        if protocol_filename != f"{label}_protocol.json":
            raise ValueError(f"{context} protocol manifest filename mismatch")
        declared_protocol_sha256 = _require_sha256(
            subject.get("protocol_manifest_sha256"),
            f"{context} protocol_manifest_sha256",
        )
        protocol_capability = capabilities[f"protocol_manifest.{label}"]
        if protocol_capability.payload_sha256 != declared_protocol_sha256:
            raise ValueError(f"{label} protocol manifest SHA-256 mismatch")

        declared_source_sha256 = _require_sha256(
            subject.get("source_manifest_sha256"),
            f"{context} source_manifest_sha256",
        )
        if declared_source_sha256 != source_oracle.manifest_sha256:
            raise ValueError(f"{label} registry source manifest SHA-256 mismatch")
        source_capability = capabilities[f"source_manifest.{label}"]
        if source_capability.payload_sha256 != declared_source_sha256:
            raise ValueError(f"{label} source manifest registry binding mismatch")

        declared_source_path = Path(
            _string(
                subject.get("source_manifest_path"),
                f"{context} source_manifest_path",
            )
        )
        if not declared_source_path.is_absolute():
            declared_source_path = inputs.repository_root / declared_source_path
        _reject_forbidden_path(declared_source_path)
        expected_source_path = (
            inputs.source_manifest_dir / f"{label}_train.json"
        )
        if _normalized(declared_source_path) != _normalized(expected_source_path):
            raise ValueError(f"{label} source manifest registry path mismatch")
    return {
        "manifest_summary": {
            "path": str(
                _normalized(inputs.protocol_manifest_dir / "manifest_summary.json")
            ),
            "sha256": oracles.manifest_summary_file_sha256,
        },
        "split_assignment": {
            "path": str(
                _normalized(inputs.protocol_manifest_dir / "split_assignment.json")
            ),
            "payload_sha256": oracles.split_assignment_payload_sha256,
            "sha256": oracles.split_assignment_file_sha256,
        },
    }


def _validate_sources(
    inputs: ProvenanceInputs,
    capabilities: Mapping[str, VerifiedArtifact],
    oracles: ProvenanceOracles,
) -> tuple[tuple[SourceTrainPt, ...], list[object]]:
    canonical_bytes: bytes | None = None
    canonical_records: list[object] | None = None
    result: list[SourceTrainPt] = []
    for source_oracle in oracles.source_files:
        subject = source_oracle.subject_id
        label = f"sub-{subject:02d}"
        manifest_path = inputs.source_manifest_dir / f"{label}_train.json"
        manifest_cap = capabilities[f"source_manifest.{label}"]
        if manifest_cap.payload_sha256 != source_oracle.manifest_sha256:
            raise ValueError(f"{label} source manifest SHA-256 mismatch")
        payload = _read_json(manifest_cap, f"{label} source manifest")
        _strict_source_manifest(payload, subject, manifest_path)
        records = payload["records"]
        if not isinstance(records, list):
            raise ValueError(f"{label} records must be an array")
        records_bytes = canonical_json_bytes(records)
        records_digest = hashlib.sha256(records_bytes).hexdigest()
        if records_digest != oracles.canonical_records_sha256:
            raise ValueError(f"{label} canonical records SHA-256 mismatch")
        if payload.get("records_sha256") != records_digest:
            raise ValueError(f"{label} declared records SHA-256 mismatch")
        if len(records) != oracles.record_count:
            raise ValueError(f"{label} source record count mismatch")
        _validate_records(records, oracles)
        if canonical_bytes is None:
            canonical_bytes = records_bytes
            canonical_records = records
        elif records_bytes != canonical_bytes:
            raise ValueError(f"{label} source record order/content mismatch")

        source_path = (
            inputs.data_root
            / "Preprocessed_data_250Hz_whiten"
            / label
            / "train.pt"
        )
        declared_source = Path(
            _string(payload.get("source_pt"), f"{label} source_pt")
        )
        _reject_forbidden_path(declared_source)
        if _normalized(declared_source) != _normalized(source_path):
            raise ValueError(f"{label} source train.pt path mismatch")
        source_cap = capabilities[f"source_train_pt.{label}"]
        if source_cap.size != source_oracle.byte_count:
            raise ValueError(f"{label} source train.pt byte count mismatch")
        if source_cap.payload_sha256 != source_oracle.sha256:
            raise ValueError(f"{label} source train.pt SHA-256 mismatch")
        result.append(
            SourceTrainPt(
                subject_id=subject,
                manifest_path=_normalized(manifest_path),
                manifest_sha256=manifest_cap.payload_sha256,
                source_path=_normalized(source_path),
                byte_count=source_cap.size,
                sha256=source_cap.payload_sha256,
            )
        )
    if canonical_records is None:  # pragma: no cover - oracle invariant
        raise AssertionError("source oracle list cannot be empty")
    return tuple(result), canonical_records


def _strict_source_manifest(
    value: Mapping[str, object],
    subject: int,
    path: Path,
) -> None:
    expected_keys = {
        "ch_names",
        "eeg_dtype",
        "eeg_shape",
        "records",
        "records_sha256",
        "schema_version",
        "source_pt",
        "split",
        "subject_id",
        "validation_concepts",
        "validation_salt",
    }
    if set(value) != expected_keys:
        raise ValueError(f"sub-{subject:02d} source manifest key mismatch")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise ValueError("source manifest schema_version must be 1")
    if value["split"] != "train":
        raise ValueError("source manifest split must be train")
    if value["subject_id"] not in (subject, f"sub-{subject:02d}"):
        raise ValueError("source manifest subject_id mismatch")
    if path.name != f"sub-{subject:02d}_train.json":
        raise ValueError("source manifest filename mismatch")
    shape = value["eeg_shape"]
    if not isinstance(shape, list) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError("source manifest eeg_shape is invalid")
    records = value["records"]
    if shape and isinstance(records, list) and shape[0] != len(records):
        raise ValueError("source manifest eeg_shape[0] mismatch")


def _validate_records(
    records: Sequence[object],
    oracles: ProvenanceOracles,
) -> None:
    concepts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(records):
        record = _mapping(raw_record, f"records[{index}]")
        required = {
            "concept_id",
            "image_id",
            "image_path",
            "row_index",
            "validation_query",
        }
        if set(record) != required:
            raise ValueError(f"records[{index}] key mismatch")
        concept = _string(record["concept_id"], f"records[{index}].concept_id")
        image = _string(record["image_id"], f"records[{index}].image_id")
        image_path = _string(
            record["image_path"], f"records[{index}].image_path"
        )
        if any(
            component.lower() == "test_images"
            for component in Path(image_path).parts
        ):
            raise ValueError("source record references forbidden test_images")
        if record["row_index"] != index or type(record["row_index"]) is not int:
            raise ValueError("source row indices must be contiguous")
        if type(record["validation_query"]) is not bool:
            raise ValueError("source validation_query must be boolean")
        pair = (concept, image)
        if pair in seen_pairs:
            raise ValueError("duplicate concept/image record")
        seen_pairs.add(pair)
        concepts[concept] += 1
    if len(concepts) != oracles.concept_count:
        raise ValueError("source concept count mismatch")
    if set(concepts.values()) != {oracles.stimuli_per_concept}:
        raise ValueError("source stimuli-per-concept mismatch")


def _validate_protocol_capabilities(
    capabilities: Mapping[str, VerifiedArtifact],
) -> None:
    for subject in _SUBJECTS:
        key = f"protocol_manifest.sub-{subject:02d}"
        capability = capabilities[key]
        if capability.artifact.role != "train":
            raise ValueError(f"{key} must select only the train role")


def _validate_model_files(
    capabilities: Mapping[str, VerifiedArtifact],
    oracles: ProvenanceOracles,
) -> None:
    expected = {
        "internvit.config": oracles.internvit_config_sha256,
        "internvit.preprocessor": oracles.internvit_preprocessor_sha256,
        "internvit.modeling": oracles.internvit_modeling_sha256,
        "internvit.weight.1": oracles.internvit_weight_sha256[0],
        "internvit.weight.2": oracles.internvit_weight_sha256[1],
        "internvit.weight.3": oracles.internvit_weight_sha256[2],
        "clip.config": oracles.clip_config_sha256,
        "clip.weights": oracles.clip_weights_sha256,
    }
    for key, digest in expected.items():
        capability = capabilities[key]
        if capability.payload_sha256 != digest:
            raise ValueError(f"{key} SHA-256 mismatch")


def _validate_caches(
    capabilities: Mapping[str, VerifiedArtifact],
    inputs: ProvenanceInputs,
    internvit: Mapping[str, object],
    oracles: ProvenanceOracles,
) -> dict[str, object]:
    descriptors = {
        "internvit_selected": (
            capabilities["cache.internvit_selected"],
            oracles.selected_cache_sha256,
            oracles.selected_cache_size,
            oracles.selected_cache_shape,
        ),
        "internvit_merged": (
            capabilities["cache.internvit_merged"],
            oracles.merged_cache_sha256,
            None,
            oracles.merged_cache_shape,
        ),
        "clip_train": (
            capabilities["cache.clip_train"],
            oracles.clip_cache_sha256,
            oracles.clip_cache_size,
            oracles.clip_cache_shape,
        ),
    }
    for key, (capability, digest, byte_count, shape) in descriptors.items():
        if capability.payload_sha256 != digest:
            raise ValueError(f"{key} cache SHA-256 mismatch")
        if byte_count is not None and capability.size != byte_count:
            raise ValueError(f"{key} cache byte count mismatch")
        header = _read_npy_header(capability, key)
        if header != (tuple(shape), False, "float16"):
            raise ValueError(
                f"{key} cache header mismatch: {header!r}"
            )

    selected_meta = _read_json(
        capabilities["cache.internvit_selected_metadata"],
        "InternViT selected cache metadata",
    )
    merged_meta = _read_json(
        capabilities["cache.internvit_merged_metadata"],
        "InternViT merged cache metadata",
    )
    clip_meta = _read_json(
        capabilities["cache.clip_train_metadata"],
        "CLIP train cache metadata",
    )
    _validate_cache_metadata(
        selected_meta,
        digest=oracles.selected_cache_sha256,
        shape=oracles.selected_cache_shape,
        context="InternViT selected cache metadata",
        digest_keys=("feature_sha256", "cache_sha256"),
        semantics={
            "layer_route": "idx0",
            "source_axes": [0, 2, 4, 6, 8],
            "pooling": "patch_mean",
            "normalization": "none",
        },
    )
    _validate_cache_metadata(
        merged_meta,
        digest=oracles.merged_cache_sha256,
        shape=oracles.merged_cache_shape,
        context="InternViT merged cache metadata",
        digest_keys=("feature_sha256", "cache_sha256"),
        semantics={},
    )
    _validate_cache_metadata(
        clip_meta,
        digest=oracles.clip_cache_sha256,
        shape=oracles.clip_cache_shape,
        context="CLIP train cache metadata",
        digest_keys=("cache_sha256", "feature_sha256"),
        semantics={"layers": [4, 6, 8, 10, 12]},
    )

    task = _mapping(internvit["task"], "InternViT task")
    if task.get("image_l2_normalization") is not True:
        raise ValueError("downstream image L2 normalization must be true")
    cache = _mapping(internvit["cache"], "InternViT cache")
    if cache.get("normalization") != "none":
        raise ValueError("extractor normalization must be none")

    return {
        "clip_train": {
            "byte_count": capabilities["cache.clip_train"].size,
            "dtype": "float16",
            "layers": [4, 6, 8, 10, 12],
            "path": str(_normalized(inputs.clip_train_cache)),
            "sha256": oracles.clip_cache_sha256,
            "shape": list(oracles.clip_cache_shape),
            "split": "train",
        },
        "internvit_merged": {
            "dtype": "float16",
            "path": str(_normalized(inputs.feature_directory / "patch_mean.npy")),
            "sha256": oracles.merged_cache_sha256,
            "shape": list(oracles.merged_cache_shape),
            "split": "train",
        },
        "internvit_selected": {
            "byte_count": capabilities["cache.internvit_selected"].size,
            "dtype": "float16",
            "path": str(_normalized(inputs.canonical_cache)),
            "sha256": oracles.selected_cache_sha256,
            "shape": list(oracles.selected_cache_shape),
            "split": "train",
        },
    }


def _validate_cache_metadata(
    metadata: Mapping[str, object],
    *,
    digest: str,
    shape: tuple[int, int, int],
    context: str,
    digest_keys: tuple[str, ...],
    semantics: Mapping[str, object],
) -> None:
    if metadata.get("split") != "train":
        raise ValueError(f"{context} split must be train")
    present_digest_keys = [key for key in digest_keys if key in metadata]
    if not present_digest_keys:
        raise ValueError(f"{context} is missing a cache digest")
    if any(metadata[key] != digest for key in present_digest_keys):
        raise ValueError(f"{context} cache digest mismatch")
    modern_keys = {"shape", "dtype", *semantics}
    present_modern = modern_keys.intersection(metadata)
    if present_modern:
        missing_modern = modern_keys - set(metadata)
        if missing_modern:
            raise ValueError(
                f"{context} is missing modern metadata fields: "
                f"{sorted(missing_modern)}"
            )
        if metadata["shape"] != list(shape):
            raise ValueError(f"{context} shape mismatch")
        if metadata["dtype"] != "float16":
            raise ValueError(f"{context} dtype mismatch")
        for key, expected in semantics.items():
            if metadata[key] != expected:
                raise ValueError(f"{context} {key} mismatch")
    else:
        # Explicit legacy train-cache schema.  It binds the cache and source
        # manifest digests, while extraction semantics are supplied by the
        # independently sealed Task 1 semantic config.
        if "manifest_sha256" not in metadata:
            raise ValueError(
                f"{context} is neither complete modern metadata nor the "
                "accepted legacy manifest-bound schema"
            )
        _require_sha256(
            metadata["manifest_sha256"],
            f"{context} legacy manifest_sha256",
        )
    _reject_forbidden_metadata(metadata, context)


def _reject_forbidden_metadata(value: object, context: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.lower() in {
                "test",
                "test_path",
                "test_cache",
                "formal_test",
                "val_confirm",
            }:
                raise ValueError(f"{context} contains forbidden scope metadata")
            _reject_forbidden_metadata(child, context)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_metadata(child, context)
    elif isinstance(value, str):
        lowered = value.lower()
        if (
            _FORMAL_TEST_RECORD_SHA256 in lowered
            or any(
                part.lower() in {"test_images", "val-confirm", "formal-test"}
                for part in Path(value).parts
            )
        ):
            raise ValueError(f"{context} contains forbidden scope metadata")


def _read_npy_header(
    capability: VerifiedArtifact,
    context: str,
) -> tuple[tuple[int, ...], bool, str]:
    with capability.open_verified() as file_object:
        try:
            version = np.lib.format.read_magic(file_object)
            if version == (1, 0):
                shape, fortran_order, dtype = (
                    np.lib.format.read_array_header_1_0(file_object)
                )
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = (
                    np.lib.format.read_array_header_2_0(file_object)
                )
            else:
                raise ValueError(f"unsupported NumPy format version {version!r}")
        except (EOFError, OSError, ValueError) as exc:
            raise ValueError(f"{context} has an invalid NumPy header") from exc
    return tuple(int(value) for value in shape), bool(fortran_order), str(dtype)


def _read_json(
    capability: VerifiedArtifact,
    context: str,
) -> dict[str, object]:
    with capability.open_verified() as file_object:
        raw = file_object.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{context} exceeds the JSON size limit")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{context} is not valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{context} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _reject_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> object:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{context} must be an integer")
    return value


def _normalized(path: Path) -> Path:
    return Path(os.path.abspath(os.path.normpath(os.fspath(path))))
