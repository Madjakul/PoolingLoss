# tests/test_pairwise_ce_loss.py

import pytest
import torch

from pooling_loss.modules import PairwiseCELoss
from pooling_loss.utils.configs import BaseConfig

BATCH_SIZE = 2
SEQUENCE_LENGTH = 128
HIDDEN_DIM = 768


@pytest.fixture
def q_embs():
    return torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_DIM)


@pytest.fixture
def k_embs():
    return torch.randn(2 * BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_DIM)


@pytest.fixture
def q_mask():
    mask = torch.ones(BATCH_SIZE, SEQUENCE_LENGTH)
    mask[:, 96:] = 0
    return mask


@pytest.fixture
def k_mask():
    mask = torch.ones(2 * BATCH_SIZE, SEQUENCE_LENGTH)
    mask[:, 96:] = 0
    return mask


def test_li_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "li"
    cfg.data.max_length = SEQUENCE_LENGTH
    cfg.train.tau = 0.07
    loss_fn = PairwiseCELoss(cfg)

    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss_metrics["loss"]), "Pairwise CE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (
        BATCH_SIZE,
        2 * BATCH_SIZE,
    ), "All scores shape mismatch"
    assert (
        ((loss_metrics["all_scores"] >= -96) & (loss_metrics["all_scores"] <= 96))
        .all()
        .item()
    ), "All scores out of expected range"
    assert loss_metrics["targets"].shape == (BATCH_SIZE,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (BATCH_SIZE,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (BATCH_SIZE,), "Negative scores shape mismatch"


def test_stabilized_li_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "average_li"
    cfg.data.max_length = SEQUENCE_LENGTH
    cfg.train.tau = 0.07
    loss_fn = PairwiseCELoss(cfg)

    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss_metrics["loss"]), "Pairwise CE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (
        BATCH_SIZE,
        2 * BATCH_SIZE,
    ), "All scores shape mismatch"
    assert (
        ((loss_metrics["all_scores"] >= 0) & (loss_metrics["all_scores"] <= 1))
        .all()
        .item()
    ), "All scores out of expected range"
    assert loss_metrics["targets"].shape == (BATCH_SIZE,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (BATCH_SIZE,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (BATCH_SIZE,), "Negative scores shape mismatch"


def test_dynamic_mean_li_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "mean_li"
    cfg.train.tau = 0.07
    cfg.data.max_length = SEQUENCE_LENGTH
    loss_fn = PairwiseCELoss(cfg)

    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss_metrics["loss"]), "Pairwise CE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (
        BATCH_SIZE,
        2 * BATCH_SIZE,
    ), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (BATCH_SIZE,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (BATCH_SIZE,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (BATCH_SIZE,), "Negative scores shape mismatch"


def test_dynamic_li_median_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "median_li"
    cfg.train.tau = 0.07
    cfg.data.max_length = SEQUENCE_LENGTH
    loss_fn = PairwiseCELoss(cfg)

    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss_metrics["loss"]), "Pairwise CE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (
        BATCH_SIZE,
        2 * BATCH_SIZE,
    ), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (BATCH_SIZE,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (BATCH_SIZE,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (BATCH_SIZE,), "Negative scores shape mismatch"


def test_dynamic_li_quantile_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "quantile_li"
    cfg.train.tau = 0.07
    cfg.data.max_length = SEQUENCE_LENGTH
    cfg.model.q = 0.75
    loss_fn = PairwiseCELoss(cfg)

    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss_metrics["loss"]), "Pairwise CE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (
        BATCH_SIZE,
        2 * BATCH_SIZE,
    ), "All scores shape mismatch"
    assert loss_metrics["targets"].shape == (BATCH_SIZE,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (BATCH_SIZE,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (BATCH_SIZE,), "Negative scores shape mismatch"


def test_mean_pooling_forward(q_embs, k_embs, q_mask, k_mask):
    cfg = BaseConfig()
    cfg.model.pooling_method = "mean"
    cfg.train.tau = 0.07
    loss_fn = PairwiseCELoss(cfg)

    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss_metrics["loss"]), "Pairwise CE loss is NaN"
    assert loss_metrics["loss"] >= 0
    assert loss_metrics["all_scores"].shape == (
        BATCH_SIZE,
        2 * BATCH_SIZE,
    ), "All scores shape mismatch"
    assert (
        ((loss_metrics["all_scores"] >= -1) & (loss_metrics["all_scores"] <= 1))
        .all()
        .item()
    ), "All scores out of expected range"
    assert loss_metrics["targets"].shape == (BATCH_SIZE,), "Targets shape mismatch"
    assert loss_metrics["poss"].shape == (BATCH_SIZE,), "Positive scores shape mismatch"
    assert loss_metrics["negs"].shape == (BATCH_SIZE,), "Negative scores shape mismatch"
