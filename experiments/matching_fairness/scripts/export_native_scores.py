from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from matching_fairness.config import Protocol  # noqa: E402
from matching_fairness.native_export import (  # noqa: E402
    NativeExportConfig,
    audit_native_checkpoints,
    export_native_scores,
)


DEFAULT_PROTOCOL = EXPERIMENT_ROOT / "configs/protocol_sub08_seed42.json"


def native_export_config_from_protocol(
    *,
    protocol: Protocol,
    source_checkout: Path,
    source_lock: Path,
    test_eeg: Path,
    test_features: Path,
    test_images: Path,
    trial_manifest: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    model: str,
    device: str,
) -> NativeExportConfig:
    protocol.assert_formal_scope()
    training = protocol.native_training
    if training.get("logit_scale_type") != "exp":
        raise ValueError("formal native export requires logit_scale_type=exp")
    if training.get("avg_trials") is not True:
        raise ValueError("formal native export requires avg_trials=true")
    return NativeExportConfig(
        source_checkout=source_checkout,
        source_lock=source_lock,
        test_eeg=test_eeg,
        test_features=test_features,
        test_images=test_images,
        trial_manifest=trial_manifest,
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        model=model,
        subject=protocol.subject,
        device=device,
        expected_image_count=200,
        n_chans=int(training["n_chans"]),
        n_times=int(training["n_times"]),
        logit_scale_type=str(training["logit_scale_type"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export or audit sealed validation-selected NICE/ATM-S scores"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--test-eeg", type=Path, required=True)
    parser.add_argument("--test-features", type=Path, required=True)
    parser.add_argument("--test-images", type=Path, required=True)
    parser.add_argument("--trial-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("nice", "atm_s"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=("main", "audit"), default="main")
    parser.add_argument(
        "--formal-artifact",
        type=Path,
        action="append",
        default=[],
        help="One completed formal ScoreArtifact directory; audit requires nine.",
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    protocol = Protocol.load(arguments.protocol)
    config = native_export_config_from_protocol(
        protocol=protocol,
        source_checkout=arguments.source_checkout,
        source_lock=arguments.source_lock,
        test_eeg=arguments.test_eeg,
        test_features=arguments.test_features,
        test_images=arguments.test_images,
        trial_manifest=arguments.trial_manifest,
        checkpoint_dir=arguments.checkpoint_dir,
        output_dir=arguments.output_dir,
        model=arguments.model,
        device=arguments.device,
    )
    if arguments.mode == "main":
        if arguments.formal_artifact:
            raise ValueError("main mode does not accept --formal-artifact")
        result = export_native_scores(config)
        payload = {
            "mode": "main",
            "artifact_paths": {
                key: str(value) for key, value in result.artifact_paths.items()
            },
            "artifact_hashes": dict(result.artifact_hashes),
        }
    else:
        if len(arguments.formal_artifact) != 9:
            raise ValueError("audit mode requires exactly nine --formal-artifact paths")
        audit_path = audit_native_checkpoints(config, arguments.formal_artifact)
        payload = {"mode": "audit", "audit_path": str(audit_path)}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
