from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator

from utils.utils_training import backward_and_clip_gradients


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _gradient_norm(model: torch.nn.Module) -> float:
    gradients = [
        parameter.grad.detach().norm()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    return float(torch.linalg.vector_norm(torch.stack(gradients)))


def test_backward_clips_the_complete_accumulated_gradient() -> None:
    accelerator = Accelerator(
        cpu=True,
        mixed_precision="no",
        gradient_accumulation_steps=2,
    )
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(10.0)
    model = accelerator.prepare(model)

    observed_norms: list[float] = []
    for _ in range(2):
        with accelerator.accumulate(model):
            loss = model(torch.tensor([[10.0]])).square().mean()
            backward_and_clip_gradients(
                accelerator=accelerator,
                model=model,
                loss=loss,
                max_grad_norm=1.0,
            )
            observed_norms.append(_gradient_norm(model))

    assert observed_norms[0] > 1.0
    assert observed_norms[1] == pytest.approx(1.0, abs=1e-6)


def test_train_clip_lora_uses_the_tested_backward_clip_operation() -> None:
    module = ast.parse(
        (PROJECT_ROOT / "train_clip_lora.py").read_text(encoding="utf-8")
    )
    called_functions = {
        node.func.id
        for node in ast.walk(module)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "backward_and_clip_gradients" in called_functions
