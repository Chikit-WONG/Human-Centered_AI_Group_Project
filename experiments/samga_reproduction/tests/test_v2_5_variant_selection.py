from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


REPRODUCTION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRODUCTION_ROOT))


def test_idx_plus_1_selection_keeps_router_logical_ids() -> None:
    selector = importlib.import_module("materialize_v2_5_variant")
    source = np.empty((2, 10, 3), dtype=np.float16)
    for axis in range(source.shape[1]):
        source[:, axis, :] = axis

    selected, selection = selector.select_variant_values(
        source, indexing_variant="idx_plus_1"
    )

    assert selected.shape == (2, 5, 3)
    assert selected.dtype == np.float16
    assert selected[0, :, 0].tolist() == [1.0, 3.0, 5.0, 7.0, 9.0]
    assert selection["logical_layer_ids"] == [20, 24, 28, 32, 36]
    assert selection["selected_captured_block_outputs"] == [21, 25, 29, 33, 37]
    assert selection["source_axis_indices"] == [1, 3, 5, 7, 9]
