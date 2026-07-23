from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from matching_fairness.artifacts import (  # noqa: E402
    publish_staged_directory,
    read_score_artifact,
)
from matching_fairness.provenance import sha256_file, sha256_path  # noqa: E402


DEFAULT_PROTOCOL = EXPERIMENT_ROOT / "configs/protocol_sub08_seed42.json"


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
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
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
        "--score-provenance-manifest",
        str(arguments.trial_split_manifest),
        "--score-protocol",
        str(arguments.protocol),
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


def export_brainrw_scores(
    arguments: argparse.Namespace,
    *,
    runner=subprocess.run,
) -> dict[str, dict[str, str]]:
    if arguments.subject_id != 8 or arguments.seed != 42:
        raise ValueError("formal BrainRW export requires subject 8 and seed 42")
    if arguments.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {arguments.output_dir}")
    arguments.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{arguments.output_dir.name}.tmp-",
            dir=arguments.output_dir.parent,
        )
    )
    try:
        staged_values = vars(arguments).copy()
        staged_values["output_dir"] = staging
        staged_arguments = argparse.Namespace(**staged_values)
        commands = build_evaluator_commands(staged_arguments)
        for command in commands.values():
            runner(command, check=True)

        artifacts = {
            name: read_score_artifact(staging / name)
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
                "sha256": _artifact_sha256(staging / name),
            }
            for name in commands
        }
        manifest = staging / "export_manifest.json"
        evaluator = REPOSITORY_ROOT / "scripts/evaluate_retrieval.py"
        brain_test = (
            Path(arguments.brain_directory)
            / f"sub-{arguments.subject_id:02d}"
            / "test.pt"
        )
        test_images = Path(arguments.image_directory) / "test_images"
        manifest_artifacts = {
            name: {
                "path": name,
                "sha256": _artifact_sha256(staging / name),
            }
            for name in commands
        }
        manifest_runs = {
            name: {
                "path": f"runs/{name}",
                "sha256": sha256_path(staging / "runs" / name),
            }
            for name in commands
        }
        encoded = (
            json.dumps(
                {
                    "schema_version": 1,
                    "scope": "fixed_formal_export",
                    "checkpoint_role": "fixed_formal",
                    "model_slug": "our_project",
                    "subject": "sub-08",
                    "seed": 42,
                    "artifacts": manifest_artifacts,
                    "runs": manifest_runs,
                    "inputs": {
                        "protocol_sha256": sha256_file(arguments.protocol),
                        "trial_manifest_sha256": sha256_file(
                            arguments.trial_split_manifest
                        ),
                        "brain_test_sha256": sha256_file(brain_test),
                        "evaluator_sha256": sha256_file(evaluator),
                        "test_image_tree_sha256": sha256_path(test_images),
                        "model_content_sha256": {
                            "brain_model": sha256_path(arguments.brain_model_path),
                            "vision_adapter": sha256_path(
                                arguments.vision_adapter_path
                            ),
                            "pretrained_vision_base": sha256_path(
                                arguments.pretrained_model_name_or_path
                            ),
                        },
                    },
                },
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        with manifest.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        publish_staged_directory(staging, arguments.output_dir)
        return inventory
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)


def main() -> None:
    arguments = build_parser().parse_args()
    inventory = export_brainrw_scores(arguments)
    print(json.dumps(inventory, indent=2, sort_keys=True))


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("metadata.json", "similarity.npy"):
        digest.update(name.encode("ascii"))
        digest.update(bytes.fromhex(sha256_file(path / name)))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
