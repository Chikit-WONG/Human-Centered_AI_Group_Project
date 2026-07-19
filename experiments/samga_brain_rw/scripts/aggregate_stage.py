#!/usr/bin/env python3
"""Aggregate one typed paired development grid into a gate decision."""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from samga_brain_rw.config import BootstrapConfig
from samga_brain_rw.registry import CandidateDecision
from samga_brain_rw.statistics import (
    CellMatrix,
    confirmation_gate,
    load_cell_matrix,
    load_strict_json_object,
    pilot_gate,
    validate_development_path,
    write_development_json_exclusive,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-matrix",
        type=Path,
        required=True,
        help="typed compact candidate CellMatrix JSON",
    )
    parser.add_argument(
        "--control-matrix",
        type=Path,
        required=True,
        help="typed compact strict-control CellMatrix JSON",
    )
    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=range(1, 6),
    )
    parser.add_argument(
        "--gate",
        choices=("pilot", "confirmation"),
        required=True,
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        help="explicit protocol JSON; required for confirmation bootstrap",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new canonical candidate-decision JSON (never overwritten)",
    )
    return parser


def _bootstrap_from_protocol(path: Path) -> BootstrapConfig:
    payload = load_strict_json_object(path)
    if payload.get("schema_version") != 1:
        raise ValueError("protocol config schema_version must be 1")
    raw = payload.get("bootstrap")
    if not isinstance(raw, dict):
        raise ValueError("protocol config must contain a bootstrap object")
    expected = {"quantile_method", "resampling", "samples", "seed"}
    if set(raw) != expected:
        raise ValueError("protocol bootstrap keys mismatch")
    return BootstrapConfig(
        samples=raw["samples"],  # type: ignore[arg-type]
        seed=raw["seed"],  # type: ignore[arg-type]
        resampling=raw["resampling"],  # type: ignore[arg-type]
        quantile_method=raw["quantile_method"],  # type: ignore[arg-type]
    )


def _mean(matrix: CellMatrix, field: str) -> float:
    values = [Decimal(str(getattr(cell, field))) for cell in matrix.cells]
    return float(sum(values, Decimal(0)) / Decimal(len(values)))


def _reject_output_aliases_inputs(
    output: Path,
    inputs: Sequence[Path],
) -> None:
    destination = validate_development_path(
        output,
        allowed_suffixes=frozenset({".json"}),
    )
    for path in inputs:
        source = validate_development_path(
            path,
            allowed_suffixes=frozenset({".json"}),
        )
        if destination == source:
            raise ValueError("aggregate output must differ from every input")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    input_paths = [
        arguments.candidate_matrix,
        arguments.control_matrix,
    ]
    if arguments.protocol_config is not None:
        input_paths.append(arguments.protocol_config)
    _reject_output_aliases_inputs(arguments.output, input_paths)

    candidate = load_cell_matrix(
        arguments.candidate_matrix,
        expected_role="candidate",
    )
    control = load_cell_matrix(
        arguments.control_matrix,
        expected_role="control",
    )
    if arguments.gate == "pilot":
        if arguments.protocol_config is not None:
            raise ValueError("--protocol-config is not accepted for a pilot gate")
        gate = pilot_gate(candidate, control, arguments.stage)
    else:
        if arguments.protocol_config is None:
            raise ValueError(
                "--protocol-config is required for a confirmation gate"
            )
        gate = confirmation_gate(
            candidate,
            control,
            _bootstrap_from_protocol(arguments.protocol_config),
        )
        gate = replace(gate, stage=arguments.stage)

    decision = CandidateDecision(
        stage=arguments.stage,
        candidate_id=candidate.config_id,
        control_id=control.config_id,
        scope=candidate.scope,  # type: ignore[arg-type]
        config_sha256=candidate.config_sha256,
        control_config_sha256=control.config_sha256,
        hyperparameters_sha256=candidate.hyperparameters_sha256,
        schedule_sha256=candidate.schedule_sha256,
        component_sha256s=candidate.component_sha256s,
        candidate_matrix_sha256=candidate.sha256,
        control_matrix_sha256=control.sha256,
        absolute_top1=_mean(candidate, "top1"),
        absolute_top5=_mean(candidate, "top5"),
        gate=gate,
    )
    write_development_json_exclusive(
        arguments.output,
        decision.to_document(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
