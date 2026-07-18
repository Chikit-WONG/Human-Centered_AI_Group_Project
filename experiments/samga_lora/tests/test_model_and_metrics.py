from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from samga_lora.model import MultiLayerCLIPProvider, SAMGALoss, SAMGATaskModel, multi_kernel_mmd
from samga_lora.utils import hash_state_dict, retrieval_metrics


def test_standard_retrieval_metrics_are_independent_topk() -> None:
    eeg = torch.tensor([[1.0, 0.0], [0.1, 0.9], [-1.0, 0.0]])
    image = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    ids = ["a", "b", "c"]
    metrics, predictions = retrieval_metrics(eeg, image, ids, ids)
    assert metrics["top1"] == 1.0
    assert metrics["top5"] == 1.0
    assert metrics["protocol"] == "standard_independent_exact_image"
    assert [row["predicted_image_id"] for row in predictions] == ids


def test_task_shapes_loss_gradients_and_shared_freeze() -> None:
    torch.manual_seed(7)
    task = SAMGATaskModel(
        layer_ids=(1, 2),
        image_dim=5,
        image_mid_dim=6,
        eeg_dim=8,
        feature_dim=4,
        channels=2,
        samples=4,
        prior_center=1,
    )
    eeg = torch.randn(3, 2, 4)
    image = torch.randn(3, 2, 5)
    subjects = torch.tensor([1, 1, 1])
    eeg_features, image_features, weights = task(eeg, image, subjects)
    assert eeg_features.shape == (3, 4)
    assert image_features.shape == (3, 4)
    assert weights.shape == (3, 2)
    loss = SAMGALoss()(eeg_features, image_features, mmd_weight=0.5)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert task.eeg_encoder.encoder.input.weight.grad is not None
    assert task.image_pre_projector.weight.grad is not None
    task.freeze_shared_encoder()
    assert not any(parameter.requires_grad for parameter in task.shared_encoder.parameters())


def test_global_router_ignores_subject_identity() -> None:
    task = SAMGATaskModel(
        layer_ids=(1, 2, 3), image_dim=5, image_mid_dim=6, eeg_dim=8,
        feature_dim=4, channels=2, samples=4, prior_center=2,
    )
    task.eval()
    image = torch.randn(2, 3, 5)
    _, weights = task.encode_image(image, torch.tensor([1, 9]), force_global=True)
    assert torch.allclose(weights[0], weights[1])


def test_mmd_zero_for_identical_features() -> None:
    values = torch.randn(8, 6)
    assert float(multi_kernel_mmd(values, values)) < 1e-6


def test_released_samga_loss_normalizes_image_only() -> None:
    eeg = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    image = torch.tensor([[4.0, 0.0], [0.0, 5.0]])
    criterion = SAMGALoss(eeg_l2norm=False, image_l2norm=True)
    output = criterion(eeg, image, mmd_weight=0.0)
    scale = F.softplus(criterion.raw_scale)
    logits = scale * eeg @ F.normalize(image, dim=-1).T
    labels = torch.arange(2)
    expected = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    assert torch.allclose(output.contrastive, expected)
    both_normalized = scale * F.normalize(eeg, dim=-1) @ F.normalize(image, dim=-1).T
    both_loss = 0.5 * (
        F.cross_entropy(both_normalized, labels) + F.cross_entropy(both_normalized.T, labels)
    )
    assert not torch.allclose(output.contrastive, both_loss)


class _DummyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.vision_model = SimpleNamespace(post_layernorm=nn.Identity())

    def forward(self, pixel_values: torch.Tensor, **_: object) -> SimpleNamespace:
        hidden = pixel_values.unsqueeze(1) * self.weight
        return SimpleNamespace(hidden_states=(hidden, hidden + 1, hidden + 2))


def test_trainable_provider_respects_outer_no_grad() -> None:
    provider = MultiLayerCLIPProvider(_DummyVision(), (1, 2), trainable=True)
    pixels = torch.randn(2, 3)
    assert provider(pixels).requires_grad
    with torch.no_grad():
        output = provider(pixels)
    assert not output.requires_grad


def test_state_hash_is_stable_and_sensitive() -> None:
    first = {"weight": torch.tensor([[1.0, 2.0]])}
    second = {"weight": torch.tensor([[1.0, 2.0]])}
    assert hash_state_dict(first) == hash_state_dict(second)
    second["weight"][0, 0] = 3.0
    assert hash_state_dict(first) != hash_state_dict(second)
