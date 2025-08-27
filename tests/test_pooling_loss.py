# tests/test_pooling_loss.py

from unittest.mock import MagicMock, patch

import lightning as L
import pytest
import torch

from pooling_loss.modules import InfoNCELoss, PoolingLoss, TripletLoss
from pooling_loss.utils.configs import BaseConfig


@pytest.fixture
def cfg():
    """Provides a default configuration."""
    config = BaseConfig()
    config.data.batch_size = 2
    config.data.max_length = 64
    config.model.base_model_name = "prajjwal1/bert-tiny"
    config.train.loss = "info-nce"
    config.train.tau = 0.07
    config.train.lr = 1e-4
    config.train.weight_decay = 1e-2
    return config


@pytest.fixture
def batch():
    """Provides a sample batch of data."""
    batch_size = 2
    seq_len = 64
    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "pos_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "pos_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "neg_input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "neg_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "labels": torch.randint(0, 1000, (batch_size, seq_len)),
    }


def test_init(cfg):
    """Tests the initialization of the PoolingLoss model."""
    model = PoolingLoss(cfg)
    assert isinstance(model, L.LightningModule)
    assert model.cfg == cfg
    assert isinstance(model.contrastive_loss, InfoNCELoss)

    # Test with TripletLoss
    cfg.train.loss = "triplet"
    cfg.train.margin = 0.5
    model = PoolingLoss(cfg)
    assert isinstance(model.contrastive_loss, TripletLoss)


def test_forward(cfg, batch):
    """Tests the forward pass of the model."""
    model = PoolingLoss(cfg)
    last_hidden_states = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    assert last_hidden_states.shape == (
        cfg.data.batch_size,
        cfg.data.max_length,
        model.hidden_size,
    )


@patch("pooling_loss.modules.modeling_pooling_loss.PoolingLoss.log_dict")
def test_training_step(mock_log_dict, cfg, batch):
    """Tests a single training step."""
    model = PoolingLoss(cfg)
    loss = model.training_step(batch, 0)
    print(loss)

    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    mock_log_dict.assert_called()
    logged_data = mock_log_dict.call_args[0][0]
    assert "alignment_loss" in logged_data
    assert "uniformity_loss" in logged_data


@patch("pooling_loss.modules.modeling_pooling_loss.PoolingLoss.log_dict")
def test_validation_step(mock_log_dict, cfg, batch):
    """Tests a single validation step."""
    model = PoolingLoss(cfg)
    model.val_auroc = MagicMock()
    model.val_hr1 = MagicMock()
    model.val_hr5 = MagicMock()
    model.val_hr10 = MagicMock()
    model.val_rr = MagicMock()

    model.validation_step(batch, 0)

    model.val_auroc.update.assert_called_once()
    model.val_hr1.update.assert_called_once()
    model.val_hr5.update.assert_called_once()
    model.val_hr10.update.assert_called_once()
    model.val_rr.update.assert_called_once()
    mock_log_dict.assert_called()


@patch("pooling_loss.modules.modeling_pooling_loss.PoolingLoss.log_dict")
def test_test_step(mock_log_dict, cfg, batch):
    """Tests a single test step."""
    model = PoolingLoss(cfg)
    model.test_auroc = MagicMock()
    model.test_hr1 = MagicMock()
    model.test_hr5 = MagicMock()
    model.test_hr10 = MagicMock()
    model.test_rr = MagicMock()

    model.test_step(batch, 0)

    model.test_auroc.update.assert_called_once()
    model.test_hr1.update.assert_called_once()
    model.test_hr5.update.assert_called_once()
    model.test_hr10.update.assert_called_once()
    model.test_rr.update.assert_called_once()
    mock_log_dict.assert_called()


def test_configure_optimizers(cfg):
    """Tests the optimizer and scheduler configuration."""
    model = PoolingLoss(cfg)
    # Mock the trainer attribute that is accessed in configure_optimizers
    model.trainer = MagicMock()
    model.trainer.estimated_stepping_batches = 1000

    optimizers = model.configure_optimizers()

    assert "optimizer" in optimizers
    assert "lr_scheduler" in optimizers
    assert isinstance(optimizers["optimizer"], torch.optim.AdamW)
    assert optimizers["optimizer"].defaults["lr"] == cfg.train.lr
    assert optimizers["lr_scheduler"]["interval"] == "step"
