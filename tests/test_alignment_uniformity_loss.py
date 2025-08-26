# tests/test_alignement_uniformity_loss.py

import pytest
import torch

from pooling_loss.modules import AlignmentUniformityLoss

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
    loss_fn = AlignmentUniformityLoss()

    loss_metrics = loss_fn(q_embs, k_embs, q_mask, k_mask)

    assert not torch.isnan(loss_metrics["alignment_loss"]), "Alignment loss is NaN"
    assert not torch.isnan(loss_metrics["uniformity_loss"]), "Uniformity loss is NaN"
    assert loss_metrics["alignment_loss"] >= 0, "Alignment loss is negative"
    assert loss_metrics["uniformity_loss"] >= 0, "Uniformity loss is negative"
