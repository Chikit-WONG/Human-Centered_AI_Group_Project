from __future__ import annotations

import inspect
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from samga_brain_rw.model import SAMGABaseConfig, SAMGATaskModel
from samga_brain_rw.upstream_samga import (
    UpstreamComponents,
    load_locked_upstream_components,
)


PINNED_COMMIT = "1a63745b7ff6f98dad34b0f0b8246a9b5260d9c1"


@pytest.fixture(scope="module")
def upstream_root(experiment_root: Path) -> Path:
    config = json.loads(
        (experiment_root / "configs" / "internvit_baseline_v1.json").read_bytes()
    )
    return Path(config["upstream"]["path"])


@pytest.fixture(scope="module")
def components(upstream_root: Path) -> UpstreamComponents:
    return load_locked_upstream_components(upstream_root, PINNED_COMMIT)


@pytest.fixture()
def task(components: UpstreamComponents) -> SAMGATaskModel:
    torch.manual_seed(20260719)
    return SAMGATaskModel(components=components, config=SAMGABaseConfig())


def test_loader_verifies_clean_pinned_checkout_and_imports_no_cli(
    upstream_root: Path,
    components: UpstreamComponents,
) -> None:
    assert components.upstream_root == upstream_root.resolve()
    assert components.commit == PINNED_COMMIT
    assert set(components.source_sha256) == {
        "module/eeg_encoder/model.py",
        "module/loss.py",
        "module/projector.py",
    }
    for component in (
        components.EEGProject,
        components.ProjectorLinear,
        components.ShareEncoder,
        components.SubjectAwareLayerMixer,
        components.ContrastiveLoss,
    ):
        source = Path(inspect.getsourcefile(component) or "")
        assert source.name != "train.py"
        assert source.is_relative_to(upstream_root.resolve())
    with pytest.raises(ValueError, match="commit"):
        load_locked_upstream_components(upstream_root, "0" * 40)


def test_locked_base_config_has_exact_internvit_contract() -> None:
    config = SAMGABaseConfig()
    assert config.layer_ids == (20, 24, 28, 32, 36)
    assert (config.image_dim, config.prior_center) == (3_200, 28)
    assert (config.image_mid_dim, config.eeg_dim, config.feature_dim) == (
        1_024,
        1_024,
        512,
    )
    assert (config.channels, config.samples) == (17, 250)
    assert config.router_eval_mode == "global"
    assert config.image_l2norm and not config.eeg_l2norm
    assert config.softplus
    with pytest.raises(ValueError, match="image_dim"):
        SAMGABaseConfig(image_dim=768)
    with pytest.raises(ValueError, match="layer_ids"):
        SAMGABaseConfig(layer_ids=(20, 24, 28))
    with pytest.raises(FrozenInstanceError):
        config.image_dim = 768  # type: ignore[misc]


def test_component_keys_outputs_and_global_router_match_upstream(
    task: SAMGATaskModel,
    components: UpstreamComponents,
) -> None:
    config = task.config
    references = {
        "eeg_encoder": components.EEGProject(
            feature_dim=config.eeg_dim,
            eeg_sample_points=config.samples,
            channels_num=config.channels,
        ),
        "eeg_projector": components.ProjectorLinear(
            config.eeg_dim, config.feature_dim
        ),
        "image_pre_projector": components.ProjectorLinear(
            config.image_dim, config.image_mid_dim
        ),
        "text_projector": components.ProjectorLinear(
            config.image_dim, config.feature_dim
        ),
        "shared_encoder": components.ShareEncoder(
            config.feature_dim, config.feature_dim
        ),
        "router": components.SubjectAwareLayerMixer(
            layer_ids=config.layer_ids,
            num_subjects=config.num_subjects,
            prior_center=config.prior_center,
            prior_strength=config.prior_strength,
            temperature=config.router_temperature,
            subject_dropout=config.router_subject_dropout,
        ),
    }
    for name, reference in references.items():
        assert tuple(getattr(task, name).state_dict()) == tuple(
            reference.state_dict()
        )
    direct_projectors = torch.nn.ModuleList(
        components.ProjectorLinear(config.image_mid_dim, config.feature_dim)
        for _ in config.layer_ids
    )
    assert tuple(task.image_projectors.state_dict()) == tuple(
        direct_projectors.state_dict()
    )

    task.eval()
    eeg = torch.randn(2, config.channels, config.samples)
    images = torch.randn(2, len(config.layer_ids), config.image_dim)
    subjects = torch.tensor([1, 10])
    eeg_output, image_output, weights = task(eeg, images, subjects)
    expected_eeg = task.shared_encoder(task.eeg_projector(task.eeg_encoder(eeg)))
    common = task.image_pre_projector(images)
    projected = torch.stack(
        [
            projector(common[:, index, :])
            for index, projector in enumerate(task.image_projectors)
        ],
        dim=1,
    )
    expected_weights = task.router(subjects, force_global=True)
    expected_image = task.shared_encoder(
        torch.sum(projected * expected_weights.unsqueeze(-1), dim=1)
    )
    assert eeg_output.shape == image_output.shape == (2, config.feature_dim)
    assert weights.shape == (2, len(config.layer_ids))
    assert torch.equal(eeg_output, expected_eeg)
    assert torch.equal(weights, expected_weights)
    assert torch.equal(image_output, expected_image)

    with torch.no_grad():
        task.router.subject_bias.weight[1, 0] = 8.0
        task.router.subject_bias.weight[10, -1] = 8.0
    _, global_weights = task.encode_image(images, subjects)
    assert torch.equal(global_weights[0], global_weights[1])
    with pytest.raises(ValueError, match="global"):
        task.encode_image(images, subjects, force_global=False)
    task.train()
    task.router.subject_dropout = 0.0
    _, subject_weights = task.encode_image(images, subjects, force_global=False)
    assert not torch.equal(subject_weights[0], subject_weights[1])


def test_loss_matches_upstream_image_only_normalization_and_mmd(
    task: SAMGATaskModel,
    components: UpstreamComponents,
) -> None:
    eeg = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 1.0]])
    image = torch.tensor([[4.0, 0.0, 2.0], [0.0, 5.0, 2.0]])
    output = task.loss(eeg, image, mmd_weight=0.25)
    scale = task.criterion.softplus(task.criterion.logit_scale)
    logits = scale * eeg @ F.normalize(image, dim=1).T
    labels = torch.arange(2)
    expected_contrastive = 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.T, labels)
    )
    expected_mmd = components.mmd_rbf(eeg, image)
    assert torch.allclose(output.contrastive, expected_contrastive)
    assert torch.allclose(output.mmd, expected_mmd)
    assert torch.allclose(
        output.total,
        0.25 * expected_mmd + 0.75 * expected_contrastive,
    )
    both_logits = (
        scale
        * F.normalize(eeg, dim=1)
        @ F.normalize(image, dim=1).T
    )
    both_loss = 0.5 * (
        F.cross_entropy(both_logits, labels)
        + F.cross_entropy(both_logits.T, labels)
    )
    assert not torch.allclose(output.contrastive, both_loss)


def test_optimizer_membership_matches_upstream_build_order(
    task: SAMGATaskModel,
) -> None:
    expected_modules = (
        task.eeg_encoder,
        task.eeg_projector,
        task.image_pre_projector,
        task.text_projector,
        task.shared_encoder,
        task.image_projectors,
        task.router,
    )
    expected = [
        parameter
        for module in expected_modules
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    actual = task.optimizer_parameters(include_shared=True)
    assert [id(value) for value in actual] == [id(value) for value in expected]
    assert len({id(value) for value in actual}) == len(actual)
    assert all(value is not task.criterion.logit_scale for value in actual)
    without_shared = task.optimizer_parameters(include_shared=False)
    shared_ids = {id(value) for value in task.shared_encoder.parameters()}
    assert not any(id(value) in shared_ids for value in without_shared)
    assert len(without_shared) + len(shared_ids) == len(actual)
    task.freeze_shared_encoder()
    assert task.shared_encoder.training is False
    assert not any(value.requires_grad for value in task.shared_encoder.parameters())
