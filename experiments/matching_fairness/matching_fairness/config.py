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
    subject: str
    seed: int
    models: tuple[ModelSpec, ...]
    standard_grid: Mapping[str, tuple[int, ...]]
    duplicate_query_counts: tuple[int, ...]
    native_training: Mapping[str, object]
    sinkhorn: Mapping[str, object]

    @classmethod
    def load(cls, path: Path) -> "Protocol":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            subject=str(payload["subject"]),
            seed=int(payload["seed"]),
            models=tuple(ModelSpec(**item) for item in payload["models"]),
            standard_grid={
                key: tuple(int(value) for value in values)
                for key, values in payload["standard_grid"].items()
            },
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
