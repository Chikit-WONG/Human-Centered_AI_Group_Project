from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from matching_fairness.artifacts import read_score_artifact  # noqa: E402
from matching_fairness.provenance import sha256_file  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export BrainRW standard/EEG-A/EEG-B ScoreArtifacts"
    )
    parser.add_argument("--brain-model-path", required=True)
    parser.add_argument("--vision-adapter-path", required=True)
    parser.add_argument("--pretrained-model-name-or-path", required=True)
    parser.add_argument("--brain-directory", required=True)
    parser.add_argument("--image-directory", required=True)
    parser.add_argument("--selected-channels", required=True)
    parser.add_argument("--trial-split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="things")
    parser.add_argument("--subject-id", type=int, default=8)
    parser.add_argument("--time-slice", default="0,250")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bf16"), default="bf16"
    )
    parser.add_argument("--cache-dir", default=".cache")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-num-samples", type=int, default=200)
    parser.add_argument("--expected-top1-count", type=int)
    parser.add_argument("--expected-top5-count", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def build_evaluator_commands(
    arguments: argparse.Namespace,
) -> dict[str, list[str]]:
    evaluator = REPOSITORY_ROOT / "scripts/evaluate_retrieval.py"
    common = [
        sys.executable,
        str(evaluator),
        "--brain-model-path",
        str(arguments.brain_model_path),
        "--vision-adapter-path",
        str(arguments.vision_adapter_path),
        "--pretrained-model-name-or-path",
        str(arguments.pretrained_model_name_or_path),
        "--brain-directory",
        str(arguments.brain_directory),
        "--image-directory",
        str(arguments.image_directory),
        "--dataset-name",
        arguments.dataset_name,
        "--subject-id",
        str(arguments.subject_id),
        "--selected-channels",
        arguments.selected_channels,
        "--time-slice",
        arguments.time_slice,
        "--batch-size",
        str(arguments.batch_size),
        "--num-workers",
        str(arguments.num_workers),
        "--device",
        arguments.device,
        "--dtype",
        arguments.dtype,
        "--cache-dir",
        arguments.cache_dir,
        "--seed",
        str(arguments.seed),
        "--expected-num-samples",
        str(arguments.expected_num_samples),
    ]
    if arguments.local_files_only:
        common.append("--local-files-only")

    commands: dict[str, list[str]] = {}
    for artifact_name, half in (
        ("standard", "standard"),
        ("eeg_a", "a"),
        ("eeg_b", "b"),
    ):
        run_dir = arguments.output_dir / "runs" / artifact_name
        command = [
            *common,
            "--trial-half",
            half,
            "--metrics-output",
            str(run_dir / "metrics.json"),
            "--predictions-output",
            str(run_dir / "predictions.csv"),
            "--score-artifact-output",
            str(arguments.output_dir / artifact_name),
        ]
        if half != "standard":
            command.extend(
                ["--trial-split-manifest", str(arguments.trial_split_manifest)]
            )
        else:
            if arguments.expected_top1_count is not None:
                command.extend(
                    ["--expected-top1-count", str(arguments.expected_top1_count)]
                )
            if arguments.expected_top5_count is not None:
                command.extend(
                    ["--expected-top5-count", str(arguments.expected_top5_count)]
                )
        commands[artifact_name] = command
    return commands


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {arguments.output_dir}")
    arguments.output_dir.mkdir(parents=True)
    commands = build_evaluator_commands(arguments)
    for command in commands.values():
        subprocess.run(command, check=True)

    artifacts = {
        name: read_score_artifact(arguments.output_dir / name)
        for name in commands
    }
    expected_shape = (
        arguments.expected_num_samples,
        arguments.expected_num_samples,
    )
    gallery_ids = artifacts["standard"].gallery_canonical_ids
    for artifact in artifacts.values():
        if artifact.similarity.shape != expected_shape:
            raise ValueError("BrainRW score artifact has an invalid matrix shape")
        if artifact.gallery_canonical_ids != gallery_ids:
            raise ValueError("BrainRW artifacts do not share canonical gallery order")
    if (
        artifacts["eeg_a"].metadata.get("query_embeddings_sha256")
        == artifacts["eeg_b"].metadata.get("query_embeddings_sha256")
    ):
        raise ValueError("BrainRW EEG-A and EEG-B query embeddings are identical")

    inventory = {
        name: {
            "path": str(arguments.output_dir / name),
            "sha256": _artifact_sha256(arguments.output_dir / name),
        }
        for name in commands
    }
    manifest = arguments.output_dir / "export_manifest.json"
    manifest.write_text(
        json.dumps(
            {"schema_version": 1, "model_slug": "our_project", "artifacts": inventory},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(inventory, indent=2, sort_keys=True))


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metadata.json", "similarity.npy"):
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(sha256_file(path / name)))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
