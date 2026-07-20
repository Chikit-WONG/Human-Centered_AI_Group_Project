from __future__ import annotations

import stat
from pathlib import Path


def test_stage2_cell_launcher_is_strict_and_development_only(
    experiment_root: Path,
) -> None:
    script = experiment_root / "scripts" / "run_stage2_cell.sh"
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\n")
    assert "set -euo pipefail" in text
    assert "HF_DATASETS_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "PYTHONPATH" in text
    assert "--scope train" in text
    assert "--validation-scope val-dev" in text
    assert "--scope val-dev" in text
    assert "--resume" in text
    assert "--subject" in text
    assert "--seed" in text
    assert "--stage 2" in text
    assert "train.py" in text
    assert "evaluate.py" in text
    lowered = text.lower()
    assert "test.pt" not in lowered
    assert "formal-test" not in lowered
    assert "val-confirm" not in lowered


def test_stage2_cell_exports_the_complete_offline_environment(
    experiment_root: Path,
) -> None:
    text = (experiment_root / "scripts" / "run_stage2_cell.sh").read_text(
        encoding="utf-8"
    )
    for variable in (
        "HF_DATASETS_OFFLINE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        assert text.count(f"export {variable}=1") == 1


def test_every_stage2_train_and_evaluation_command_seals_cuda(
    experiment_root: Path,
) -> None:
    text = (experiment_root / "scripts" / "run_stage2_cell.sh").read_text(
        encoding="utf-8"
    )
    model_commands = [
        " ".join(block.replace("\\", " ").split())
        for block in text.split("\n\n")
        if "train.py" in block or "evaluate.py" in block
    ]
    train_commands = [
        command for command in model_commands if "train.py" in command
    ]
    evaluation_commands = [
        command for command in model_commands if "evaluate.py" in command
    ]
    assert len(train_commands) == 1
    assert len(evaluation_commands) == 3
    for command in model_commands:
        tokens = command.split()
        assert tokens.count("--device") == 1
        assert tokens[tokens.index("--device") + 1] == "cuda"


def test_stage2_cell_launcher_requires_exact_positional_contract(
    experiment_root: Path,
) -> None:
    text = (experiment_root / "scripts" / "run_stage2_cell.sh").read_text(
        encoding="utf-8"
    )
    for position in range(1, 11):
        assert f"${{{position}:?" in text


def test_stage2_cell_emits_four_parity_bundles_before_completion(
    experiment_root: Path,
) -> None:
    text = (experiment_root / "scripts" / "run_stage2_cell.sh").read_text(
        encoding="utf-8"
    )
    for directory in (
        "in_loop",
        "saved_checkpoint",
        "repeat_emission",
        "reload_evaluation",
    ):
        assert f"${{OUTPUT_DIR}}/{directory}" in text
    assert text.count("evaluate.py") == 3
    assert "check_baseline_parity.py" in text
    assert "--run-dir" in text
    assert "--scope val-dev" in text
    assert "baseline_parity.json" in text
    assert "complete-env" in text
    assert "--output-hashes" in text
    assert text.index("check_baseline_parity.py") < text.index("complete-env")


def test_stage2_full_completion_emits_exact_three_locked_hashes(
    experiment_root: Path,
) -> None:
    text = (experiment_root / "scripts" / "run_stage2_cell.sh").read_text(
        encoding="utf-8"
    )
    assert "metrics_sha256" not in text
    assert (
        'sha256sum "${OUTPUT_DIR}/checkpoint_epoch060.pt"'
    ) in text
    assert 'sha256sum "${OUTPUT_DIR}/run_manifest.json"' in text
    assert 'sha256sum "${OUTPUT_DIR}/baseline_parity.json"' in text
    assert text.count('\\"final_checkpoint_sha256\\"') == 1
    assert text.count('\\"parity_sha256\\"') == 1
    assert text.count('\\"run_manifest_sha256\\"') == 1
    assert (
        '"{\\"final_checkpoint_sha256\\":\\"${FINAL_CHECKPOINT_SHA256}\\",'
        '\\"parity_sha256\\":\\"${PARITY_SHA256}\\",'
        '\\"run_manifest_sha256\\":\\"${RUN_MANIFEST_SHA256}\\"}"'
    ) in text


def test_stage2_cell_forwards_locked_optional_candidate_inputs(
    experiment_root: Path,
) -> None:
    text = (experiment_root / "scripts" / "run_stage2_cell.sh").read_text(
        encoding="utf-8"
    )
    assert "SAMGA_ADAPTER_RANK" in text
    assert "SAMGA_ADAPTER_LR_RATIO" in text
    assert "--adapter-rank" in text
    assert "--adapter-lr-ratio" in text
    assert "SAMGA_WHITENING_ARTIFACT" in text
    assert "--whitening-artifact" in text
    assert "TRAIN_EXTRA_ARGS" in text


def test_stage2_cell_launcher_is_executable(
    experiment_root: Path,
) -> None:
    script = experiment_root / "scripts" / "run_stage2_cell.sh"
    mode = script.stat().st_mode
    assert mode & stat.S_IXUSR
