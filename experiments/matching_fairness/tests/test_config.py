from pathlib import Path
import json

import pytest

from matching_fairness.config import Protocol


CONFIG = Path("experiments/matching_fairness/configs/protocol_sub08_seed42.json")


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
