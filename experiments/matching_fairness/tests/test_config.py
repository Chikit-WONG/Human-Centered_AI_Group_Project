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
