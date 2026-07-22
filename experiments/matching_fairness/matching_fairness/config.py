import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    encoder_type: str
    checkpoint_role: str


@dataclass(frozen=True)
class Protocol:
    schema_version: int
    subject: str
    seed: int
    models: tuple[ModelSpec, ...]
    standard_grid: Mapping[str, tuple[int, ...]]
    duplicate_query_counts: tuple[int, ...]
    native_training: Mapping[str, object]
    sinkhorn: Mapping[str, object]

    @classmethod
    def load(cls, path: Path) -> "Protocol":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("protocol must be valid UTF-8 JSON") from error
        expected_keys = {
            "schema_version",
            "subject",
            "seed",
            "models",
            "native_training",
            "standard_grid",
            "duplicate_query_counts",
            "sinkhorn",
        }
        if not isinstance(payload, dict) or set(payload) != expected_keys:
            raise ValueError("protocol must use the exact formal schema")
        if (
            type(payload["schema_version"]) is not int
            or not isinstance(payload["subject"], str)
            or type(payload["seed"]) is not int
            or not isinstance(payload["models"], list)
            or not isinstance(payload["native_training"], dict)
            or not isinstance(payload["standard_grid"], dict)
            or not isinstance(payload["duplicate_query_counts"], list)
            or not isinstance(payload["sinkhorn"], dict)
        ):
            raise ValueError("protocol must use the exact formal schema types")
        if payload["schema_version"] != 1:
            raise ValueError("protocol schema_version must be 1")
        model_keys = {"slug", "encoder_type", "checkpoint_role"}
        if any(
            not isinstance(item, dict)
            or set(item) != model_keys
            or any(not isinstance(value, str) for value in item.values())
            for item in payload["models"]
        ):
            raise ValueError("protocol must use the exact formal schema types")
        if any(
            not isinstance(key, str)
            or not isinstance(values, list)
            or any(type(value) is not int for value in values)
            for key, values in payload["standard_grid"].items()
        ) or any(
            type(value) is not int for value in payload["duplicate_query_counts"]
        ):
            raise ValueError("protocol must use the exact formal schema types")
        try:
            models = tuple(ModelSpec(**item) for item in payload["models"])
            standard_grid = {
                key: tuple(values)
                for key, values in payload["standard_grid"].items()
            }
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("protocol contains an invalid formal field") from error
        return cls(
            schema_version=1,
            subject=payload["subject"],
            seed=payload["seed"],
            models=models,
            standard_grid=standard_grid,
            duplicate_query_counts=tuple(payload["duplicate_query_counts"]),
            native_training=dict(payload["native_training"]),
            sinkhorn=dict(payload["sinkhorn"]),
        )

    @property
    def standard_scenario_count(self) -> int:
        return math.prod(len(values) for values in self.standard_grid.values())

    def assert_formal_scope(self) -> None:
        actual = (self.subject, self.seed, tuple(m.slug for m in self.models))
        expected = ("sub-08", 42, ("nice", "atm_s", "our_project"))
        if actual != expected:
            raise ValueError(f"formal scope must be sub-08 / seed-42: {actual}")
        if self.standard_scenario_count != 27:
            raise ValueError("formal standard grid must contain exactly 27 scenarios")
        if self.duplicate_query_counts != (0, 10, 20):
            raise ValueError("duplicate-query counts must be exactly (0, 10, 20)")
        expected_models = (
            ModelSpec("nice", "NICE", "val_selected_formal"),
            ModelSpec("atm_s", "ATMS", "val_selected_formal"),
            ModelSpec("our_project", "BrainRW", "fixed_formal"),
        )
        expected_native_training = {
            "mode": "intra",
            "epochs": 500,
            "batch_size": 1024,
            "lr": 0.0003,
            "val_ratio": 0.1,
            "early_stopping_patience": 10,
            "ema_decay": 0.999,
            "logit_scale_type": "exp",
            "avg_trials": True,
            "n_chans": 63,
            "n_times": 250,
            "checkpoint_metric": "validation_contrastive_loss",
            "checkpoint_direction": "min",
        }
        expected_standard_grid = {
            "drop_query": (0, 5, 10),
            "drop_gallery": (0, 5, 10),
            "drop_pair": (0,),
            "duplicate_gallery": (0, 10, 20),
        }
        expected_sinkhorn = {
            "temperature": 0.05,
            "max_iterations": 500,
            "tolerance": 1e-8,
        }
        if (
            self.schema_version != 1
            or self.models != expected_models
            or _canonical_json(self.native_training)
            != _canonical_json(expected_native_training)
            or _canonical_json(self.standard_grid)
            != _canonical_json(expected_standard_grid)
            or _canonical_json(self.sinkhorn) != _canonical_json(expected_sinkhorn)
        ):
            raise ValueError("protocol does not match the canonical formal protocol")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
