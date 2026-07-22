from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from matching_fairness.config import Protocol  # noqa: E402
from matching_fairness.native_training import (  # noqa: E402
    NativeTrainConfig,
    train_native,
)


DEFAULT_PROTOCOL = EXPERIMENT_ROOT / "configs/protocol_sub08_seed42.json"


def native_config_from_protocol(
    *,
    protocol: Protocol,
    source_checkout: Path,
    source_lock: Path,
    training_eeg: Path,
    training_features: Path,
    output_dir: Path,
    model: str,
) -> NativeTrainConfig:
    protocol.assert_formal_scope()
    training = protocol.native_training
    if training.get("mode") != "intra":
        raise ValueError("formal native training mode must be intra")
    if training.get("checkpoint_metric") != "validation_contrastive_loss":
        raise ValueError("formal checkpoint metric must be validation contrastive loss")
    if training.get("checkpoint_direction") != "min":
        raise ValueError("formal checkpoint direction must be min")
    return NativeTrainConfig(
        source_checkout=source_checkout,
        source_lock=source_lock,
        training_eeg=training_eeg,
        training_features=training_features,
        output_dir=output_dir,
        model=model,
        subject=protocol.subject,
        seed=protocol.seed,
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["lr"]),
        val_ratio=float(training["val_ratio"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        ema_decay=float(training["ema_decay"]),
        logit_scale_type=str(training["logit_scale_type"]),
        avg_trials=bool(training["avg_trials"]),
        n_chans=int(training["n_chans"]),
        n_times=int(training["n_times"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a sealed official NICE or ATM-S validation-selected baseline"
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--training-eeg", type=Path, required=True)
    parser.add_argument("--training-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("nice", "atm_s"), required=True)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    protocol = Protocol.load(arguments.protocol)
    config = native_config_from_protocol(
        protocol=protocol,
        source_checkout=arguments.source_checkout,
        source_lock=arguments.source_lock,
        training_eeg=arguments.training_eeg,
        training_features=arguments.training_features,
        output_dir=arguments.output_dir,
        model=arguments.model,
    )
    result = train_native(config)
    print(
        json.dumps(
            {
                "best_checkpoint": str(result.best_checkpoint),
                "history": str(result.history),
                "history_sha256": result.history_sha256,
                "manifest": str(result.manifest),
                "selected_epoch": result.selected.epoch,
                "selected_val_loss": result.selected.val_loss,
                "stopped_early": result.stopped_early,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
