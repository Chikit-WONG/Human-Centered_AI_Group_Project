from pathlib import Path
import json

import pytest
import yaml

from matching_fairness.config import Protocol


CONFIG = Path("experiments/matching_fairness/configs/protocol_sub08_seed42.json")
ENVIRONMENT = Path(
    "experiments/matching_fairness/configs/atm_native_environment.yml"
)
README = Path("experiments/matching_fairness/README.md")
README_ZH = Path("experiments/matching_fairness/README_ZH.md")


def test_atm_native_environment_uses_official_channels_and_pinned_wheels() -> None:
    payload = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))

    assert payload["name"] == "atm_native"
    assert payload["channels"] == [
        "https://conda.anaconda.org/conda-forge",
        "nodefaults",
    ]

    dependencies = payload["dependencies"]
    assert dependencies[:3] == ["python=3.12", "pip", "setuptools=75.8.0"]
    assert len(dependencies) == 4
    pip_dependencies = dependencies[3]["pip"]
    assert pip_dependencies == [
        "--index-url https://pypi.org/simple",
        "--extra-index-url https://download.pytorch.org/whl/cu124",
        "torch==2.5.0+cu124",
        "torchvision==0.20.0+cu124",
        "torchaudio==2.5.0+cu124",
        "numpy==1.26.4",
        "pandas==2.3.3",
        "scipy==1.15.3",
        "scikit-learn==1.6.1",
        "mne==1.9.0",
        "einops==0.8.1",
        "braindecode==0.8.1",
        "wandb==0.19.10",
        "open-clip-torch==2.26.1",
        "pytest==8.3.5",
        "git+https://github.com/openai/CLIP.git@"
        "a9b1bf5920416aaeaec965c25dd9e8f98c864f16",
    ]


def test_atm_native_readmes_clear_inherited_library_path_for_git() -> None:
    command = (
        "PIP_NO_BUILD_ISOLATION=1 env -u LD_LIBRARY_PATH "
        "conda env create -n atm_native"
    )
    assert command in README.read_text(encoding="utf-8")
    assert command in README_ZH.read_text(encoding="utf-8")


def test_atm_native_readmes_document_exact_commit_cache_fallback() -> None:
    cache = (
        "/hpc2hdd/home/ckwong627/workdir/new_sub_workdir/EEG_Project/models/"
        "openai_clip_a9b1bf5920416aaeaec965c25dd9e8f98c864f16_shallow"
    )
    revision = "a9b1bf5920416aaeaec965c25dd9e8f98c864f16"
    for readme in (README, README_ZH):
        text = readme.read_text(encoding="utf-8")
        assert cache in text
        assert f"git+file://{cache}@{revision}" in text


def test_formal_protocol_is_exactly_sub08_seed42() -> None:
    protocol = Protocol.load(CONFIG)
    assert protocol.subject == "sub-08"
    assert protocol.seed == 42
    assert tuple(model.slug for model in protocol.models) == (
        "nice",
        "atm_s",
        "our_project",
    )
    assert protocol.standard_scenario_count == 27
    assert protocol.duplicate_query_counts == (0, 10, 20)
    protocol.assert_formal_scope()


def test_scope_guard_rejects_multisubject(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["subject"] = "all"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sub-08 / seed-42"):
        Protocol.load(path).assert_formal_scope()


def test_protocol_load_rejects_minimal_subject_seed_json(tmp_path: Path) -> None:
    path = tmp_path / "minimal.json"
    path.write_text(
        json.dumps({"subject": "sub-08", "seed": 42}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact formal schema"):
        Protocol.load(path)


@pytest.mark.parametrize(
    "field,value",
    (
        ("schema_version", True),
        ("subject", 8),
        ("seed", 42.0),
    ),
)
def test_protocol_load_rejects_coercible_top_level_types(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "coercible.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exact formal schema types"):
        Protocol.load(path)


def test_protocol_load_rejects_coercible_standard_grid_value(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["standard_grid"]["drop_query"][0] = 0.0
    path = tmp_path / "coercible-grid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exact formal schema types"):
        Protocol.load(path)


@pytest.mark.parametrize(
    "section,key,value",
    (
        ("native_training", "n_times", 251),
        ("native_training", "n_times", 250.0),
        ("sinkhorn", "temperature", 0.1),
    ),
)
def test_scope_guard_rejects_nested_protocol_tampering(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload[section][key] = value
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical formal protocol"):
        Protocol.load(path).assert_formal_scope()
