#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from samga_lora.utils import atomic_write_json, read_json  # noqa: E402


def first_jsonl(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.loads(next(line for line in handle if line.strip()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate the W1 frozen/LoRA smoke runs")
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--parity-report", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.smoke_root)
    checks: dict[str, Any] = {}
    passed = True
    for mode in ("frozen", "lora"):
        run = root / mode
        completion = read_json(run / "completion.json")
        history = first_jsonl(run / "training_history.jsonl")
        validation = first_jsonl(run / "validation_metrics.jsonl")
        reload = read_json(run / "checkpoint_reload.json")
        gradients = history["first_step_gradient_norms"]
        mode_ok = bool(
            completion["completed"]
            and int(completion["global_step"]) == 1
            and float(gradients["task"]) > 0
            and (float(gradients["vision"]) > 0 if mode == "lora" else gradients["vision"] == 0)
            and validation["protocol"] == "standard_independent_exact_image"
            and int(validation["num_queries"]) == 200
            and reload["passed"]
        )
        checks[mode] = {
            "passed": mode_ok,
            "gradient_norms": gradients,
            "validation_top1": validation["top1"],
            "validation_top5": validation["top5"],
            "peak_cuda_memory_bytes": completion["peak_cuda_memory_bytes"],
        }
        passed = passed and mode_ok
    parity = read_json(args.parity_report)
    checks["cache_and_initialization_parity"] = parity
    passed = passed and bool(parity["passed"])
    report = {"schema_version": 1, "passed": passed, "checks": checks}
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
