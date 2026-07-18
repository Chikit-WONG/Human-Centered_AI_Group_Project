#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.data import build_subject_manifest  # noqa: E402
from samga_lora.utils import atomic_write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build immutable THINGS-EEG2 row manifests")
    parser.add_argument("--things-root", required=True)
    parser.add_argument("--brain-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--validation-concepts", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "things_root": str(Path(args.things_root).resolve()),
        "brain_root": str(Path(args.brain_root).resolve()),
        "validation_concepts": args.validation_concepts,
        "subjects": {},
    }
    reference_hashes: dict[str, str] = {}
    for subject_id in args.subjects:
        if not 1 <= subject_id <= 10:
            raise ValueError(f"Invalid subject {subject_id}; expected 1..10")
        subject_entry: dict[str, object] = {}
        for split in ("train", "test"):
            output = output_dir / f"sub-{subject_id:02d}_{split}.json"
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {output}")
            manifest = build_subject_manifest(
                things_root=args.things_root,
                brain_root=args.brain_root,
                subject_id=subject_id,
                split=split,
                output_path=output,
                validation_concepts=args.validation_concepts,
            )
            records_hash = str(manifest["records_sha256"])
            if split in reference_hashes and records_hash != reference_hashes[split]:
                raise RuntimeError(
                    f"Image row order differs across subjects for {split}: "
                    f"{records_hash} != {reference_hashes[split]}"
                )
            reference_hashes.setdefault(split, records_hash)
            subject_entry[split] = {
                "manifest": str(output),
                "rows": len(manifest["records"]),
                "records_sha256": records_hash,
                "eeg_shape": manifest["eeg_shape"],
            }
        summary["subjects"][f"sub-{subject_id:02d}"] = subject_entry  # type: ignore[index]
    summary["shared_records_sha256"] = reference_hashes
    atomic_write_json(output_dir / "manifest_summary.json", summary)


if __name__ == "__main__":
    main()
