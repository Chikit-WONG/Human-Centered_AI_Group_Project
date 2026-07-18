import json
import os
import subprocess
from pathlib import Path

import pytest


RUNNER = Path(__file__).resolve().parents[1] / "run_official_cell.sh"
PROTOCOL_ENV_NAMES = (
    "SAMGA_EEG_L2NORM",
    "SAMGA_T_LEARNABLE",
    "SAMGA_ROUTER_LAYER_DROPOUT",
)
ISOLATED_ENV_NAMES = PROTOCOL_ENV_NAMES + (
    "SAMGA_FEATURE_ROOT",
    "SAMGA_SEED",
    "SAMGA_BATCH_SIZE",
    "SAMGA_NUM_EPOCHS",
    "SAMGA_EARLY_STOP_PATIENCE",
)


def _value_after(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for name in ISOLATED_ENV_NAMES:
        env.pop(name, None)
    return env


def _prepare_fake_run(
    tmp_path: Path, *, variant: str, override_feature_root: bool = True
) -> tuple[dict[str, str], Path]:
    project_root = tmp_path / "project"
    samga_root = tmp_path / "official-samga"
    repro_root = tmp_path / "reproduction"
    eeg_root = repro_root / "eeg"
    feature_root = (
        tmp_path / "feature-cache"
        if override_feature_root
        else repro_root / "features" / variant
    )
    capture = tmp_path / "argv.json"

    (eeg_root / "sub-08").mkdir(parents=True)
    feature_root.mkdir(parents=True)
    samga_root.mkdir(parents=True)

    for path in (
        eeg_root / "info.json",
        eeg_root / "sub-08" / "train.npy",
        eeg_root / "sub-08" / "test.npy",
        feature_root / "image_train_layer20.npy",
        feature_root / "image_test_layer20.npy",
    ):
        path.touch()

    (samga_root / "train.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['SAMGA_TEST_ARGV_CAPTURE']).write_text("
        "json.dumps(sys.argv[1:]), encoding='utf-8')\n",
        encoding="utf-8",
    )

    env = _clean_env()
    env.update(
        {
            "SAMGA_PROJECT_ROOT": str(project_root),
            "SAMGA_REFERENCE_ROOT": str(samga_root),
            "SAMGA_REPRO_ROOT": str(repro_root),
            "SAMGA_EEG_ROOT": str(eeg_root),
            "SAMGA_VARIANT": variant,
            "SAMGA_TEST_ARGV_CAPTURE": str(capture),
        }
    )
    if override_feature_root:
        env["SAMGA_FEATURE_ROOT"] = str(feature_root)
    return env, capture


def _run(
    env: dict[str, str], subject: str = "8"
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(RUNNER), subject],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_default_protocol_does_not_add_optional_train_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in PROTOCOL_ENV_NAMES:
        monkeypatch.setenv(name, "1" if name != "SAMGA_ROUTER_LAYER_DROPOUT" else "0.25")
    monkeypatch.setenv("SAMGA_FEATURE_ROOT", "/ambient/feature/root/must-not-leak")
    variant = "baseline-defaults"
    env, capture = _prepare_fake_run(
        tmp_path,
        variant=variant,
        override_feature_root=False,
    )

    completed = _run(env)

    assert completed.returncode == 0, completed.stderr
    argv = json.loads(capture.read_text(encoding="utf-8"))
    expected_feature_root = Path(env["SAMGA_REPRO_ROOT"]) / "features" / variant
    assert _value_after(argv, "--image_feature_dir") == str(expected_feature_root)
    assert "--img_l2norm" in argv
    assert "--eeg_l2norm" not in argv
    assert "--t_learnable" not in argv
    assert "--router_layer_dropout" not in argv
    assert _value_after(argv, "--seed") == "2025"
    assert _value_after(argv, "--batch_size") == "1024"
    assert _value_after(argv, "--early_stop_patience") == "0"
    assert _value_after(argv, "--num_epochs") == "60"


def test_enabled_protocol_adds_exact_optional_train_flags_and_isolates_paths(
    tmp_path: Path,
) -> None:
    variant = "eeg-l2-tlearn-routerdrop025"
    env, capture = _prepare_fake_run(tmp_path, variant=variant)
    env.update(
        {
            "SAMGA_EEG_L2NORM": "1",
            "SAMGA_T_LEARNABLE": "1",
            "SAMGA_ROUTER_LAYER_DROPOUT": "0.25",
        }
    )

    completed = _run(env)

    assert completed.returncode == 0, completed.stderr
    argv = json.loads(capture.read_text(encoding="utf-8"))
    assert argv.count("--eeg_l2norm") == 1
    assert argv.count("--t_learnable") == 1
    assert argv.count("--router_layer_dropout") == 1
    assert _value_after(argv, "--router_layer_dropout") == "0.25"
    assert argv[-4:] == [
        "--eeg_l2norm",
        "--t_learnable",
        "--router_layer_dropout",
        "0.25",
    ]
    assert _value_after(argv, "--image_feature_dir") == env["SAMGA_FEATURE_ROOT"]
    assert f"/official_runs/{variant}/" in _value_after(argv, "--output_dir")
    assert variant in _value_after(argv, "--output_name")


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SAMGA_EEG_L2NORM", ""),
        ("SAMGA_EEG_L2NORM", "true"),
        ("SAMGA_EEG_L2NORM", "2"),
        ("SAMGA_T_LEARNABLE", ""),
        ("SAMGA_T_LEARNABLE", "false"),
        ("SAMGA_T_LEARNABLE", "-1"),
    ],
)
def test_boolean_protocol_values_other_than_zero_or_one_fail_early(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    capture = tmp_path / "must-not-run.json"
    env = _clean_env()
    env.update(
        {
            "SAMGA_VARIANT": "invalid-boolean",
            "SAMGA_TEST_ARGV_CAPTURE": str(capture),
            name: value,
        }
    )

    completed = _run(env)

    assert completed.returncode == 2
    assert f"{name} must be 0 or 1" in completed.stderr
    assert not capture.exists()


@pytest.mark.parametrize(
    "value",
    ["", "-0.1", "-1e-9999", "1", "1.0", "1.1", "nan", "inf", "abc"],
)
def test_router_layer_dropout_outside_finite_half_open_unit_interval_fails_early(
    tmp_path: Path,
    value: str,
) -> None:
    capture = tmp_path / "must-not-run.json"
    env = _clean_env()
    env.update(
        {
            "SAMGA_VARIANT": "invalid-dropout",
            "SAMGA_TEST_ARGV_CAPTURE": str(capture),
            "SAMGA_ROUTER_LAYER_DROPOUT": value,
        }
    )

    completed = _run(env)

    assert completed.returncode == 2
    assert "SAMGA_ROUTER_LAYER_DROPOUT must be a finite number in [0, 1)" in completed.stderr
    assert not capture.exists()


@pytest.mark.parametrize(
    "value", ["0", "0.0", "-0", "-0.0e9999", ".25", "2.5e-1", "1e-9999"]
)
def test_valid_router_layer_dropout_values_are_accepted(
    tmp_path: Path,
    value: str,
) -> None:
    env, capture = _prepare_fake_run(tmp_path, variant=f"dropout-{value}")
    env["SAMGA_ROUTER_LAYER_DROPOUT"] = value

    completed = _run(env)

    assert completed.returncode == 0, completed.stderr
    argv = json.loads(capture.read_text(encoding="utf-8"))
    if value in {"0", "0.0", "-0", "-0.0e9999"}:
        assert "--router_layer_dropout" not in argv
    else:
        assert _value_after(argv, "--router_layer_dropout") == value


@pytest.mark.parametrize("subject", ["abc", "1+1", "08", "0", "11"])
def test_subject_must_be_an_unambiguous_decimal_id(
    tmp_path: Path,
    subject: str,
) -> None:
    env = _clean_env()
    env["SAMGA_VARIANT"] = "subject-validation"

    completed = _run(env, subject)

    assert completed.returncode == 2
    assert "SUBJECT must be a decimal integer in 1..10" in completed.stderr


def test_subject_validation_does_not_evaluate_arithmetic_input(tmp_path: Path) -> None:
    marker = tmp_path / "arithmetic-command-substitution-ran"
    subject = f"x[$(touch {marker})0]"
    env = _clean_env()
    env["SAMGA_VARIANT"] = "subject-injection"

    completed = _run(env, subject)

    assert completed.returncode == 2
    assert "SUBJECT must be a decimal integer in 1..10" in completed.stderr
    assert not marker.exists()


@pytest.mark.parametrize(
    "variant",
    ["../escape", ".hidden", "contains/slash", "contains space"],
)
def test_variant_must_be_a_safe_path_slug(tmp_path: Path, variant: str) -> None:
    env = _clean_env()
    env["SAMGA_VARIANT"] = variant

    completed = _run(env)

    assert completed.returncode == 2
    assert (
        "SAMGA_VARIANT must match [A-Za-z0-9][A-Za-z0-9._-]*"
        in completed.stderr
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("SAMGA_SEED", "", "SAMGA_SEED must be an integer in 0..4294967295"),
        ("SAMGA_SEED", "-1", "SAMGA_SEED must be an integer in 0..4294967295"),
        ("SAMGA_SEED", "01", "SAMGA_SEED must be an integer in 0..4294967295"),
        (
            "SAMGA_SEED",
            "4294967296",
            "SAMGA_SEED must be an integer in 0..4294967295",
        ),
        ("SAMGA_BATCH_SIZE", "", "SAMGA_BATCH_SIZE must be a positive decimal integer"),
        ("SAMGA_BATCH_SIZE", "0", "SAMGA_BATCH_SIZE must be a positive decimal integer"),
        ("SAMGA_BATCH_SIZE", "01", "SAMGA_BATCH_SIZE must be a positive decimal integer"),
        ("SAMGA_BATCH_SIZE", "-1", "SAMGA_BATCH_SIZE must be a positive decimal integer"),
        ("SAMGA_NUM_EPOCHS", "", "SAMGA_NUM_EPOCHS must be a positive decimal integer"),
        ("SAMGA_NUM_EPOCHS", "0", "SAMGA_NUM_EPOCHS must be a positive decimal integer"),
        ("SAMGA_NUM_EPOCHS", "01", "SAMGA_NUM_EPOCHS must be a positive decimal integer"),
        (
            "SAMGA_EARLY_STOP_PATIENCE",
            "",
            "SAMGA_EARLY_STOP_PATIENCE must be a non-negative decimal integer",
        ),
        (
            "SAMGA_EARLY_STOP_PATIENCE",
            "-1",
            "SAMGA_EARLY_STOP_PATIENCE must be a non-negative decimal integer",
        ),
        (
            "SAMGA_EARLY_STOP_PATIENCE",
            "01",
            "SAMGA_EARLY_STOP_PATIENCE must be a non-negative decimal integer",
        ),
    ],
)
def test_numeric_protocol_values_are_validated_before_paths_or_assets(
    tmp_path: Path,
    name: str,
    value: str,
    message: str,
) -> None:
    env = _clean_env()
    env["SAMGA_VARIANT"] = "invalid-numeric"
    env["SAMGA_REPRO_ROOT"] = str(tmp_path / "reproduction")
    env[name] = value

    completed = _run(env)

    assert completed.returncode == 2
    assert message in completed.stderr


@pytest.mark.parametrize("name", ["SAMGA_SEED", "SAMGA_BATCH_SIZE", "SAMGA_EARLY_STOP_PATIENCE"])
def test_numeric_protocol_values_cannot_escape_the_result_tree(
    tmp_path: Path,
    name: str,
) -> None:
    env = _clean_env()
    env["SAMGA_VARIANT"] = "path-isolation"
    env["SAMGA_REPRO_ROOT"] = str(tmp_path / "reproduction")
    env[name] = "/../../../../escaped-result"

    completed = _run(env)

    assert completed.returncode == 2
    assert not (tmp_path / "escaped-result").exists()


@pytest.mark.parametrize("seed", ["0", "4294967295"])
def test_numeric_protocol_boundary_values_reach_train_argv(
    tmp_path: Path,
    seed: str,
) -> None:
    env, capture = _prepare_fake_run(tmp_path, variant=f"numeric-seed-{seed}")
    env.update(
        {
            "SAMGA_SEED": seed,
            "SAMGA_BATCH_SIZE": "1",
            "SAMGA_NUM_EPOCHS": "1",
            "SAMGA_EARLY_STOP_PATIENCE": "0",
        }
    )

    completed = _run(env)

    assert completed.returncode == 0, completed.stderr
    argv = json.loads(capture.read_text(encoding="utf-8"))
    assert _value_after(argv, "--seed") == seed
    assert _value_after(argv, "--batch_size") == "1"
    assert _value_after(argv, "--num_epochs") == "1"
    assert _value_after(argv, "--early_stop_patience") == "0"
