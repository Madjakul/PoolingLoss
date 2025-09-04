# tests/test_callbacks.py

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import lightning as L
import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel

from pooling_loss.callbacks.grad_norm_monitor import GradNormMonitor
from pooling_loss.callbacks.pe_pca import PositionalEmbeddingPCA
from pooling_loss.callbacks.pe_tracker import PositionalEmbeddingTracker

# --- Fixed Fixtures ---


class MockLanguageModel(nn.Module):
    """A mock language model that mimics the structure of a Hugging Face
    model."""

    def __init__(self, config):
        super().__init__()
        # Create a RobertaModel and add a 'roberta' attribute to match the callback expectations
        self.model = RobertaModel(config)
        # Add a 'roberta' attribute to match the callback's expected path
        self.model.roberta = self.model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MockLitModule(L.LightningModule):
    """A mock LightningModule that contains a language model and parameters."""

    def __init__(self):
        super().__init__()
        config = RobertaConfig(
            vocab_size=1000,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=128,
        )
        self.lm = MockLanguageModel(config)
        self.linear = nn.Linear(32, 2)

    def training_step(self, batch, batch_idx):
        output = self.lm(input_ids=batch["input_ids"])
        loss = self.linear(output.last_hidden_state).mean()
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


@pytest.fixture
def mock_pl_module():
    """Provides a mock LightningModule instance."""
    return MockLitModule()


@pytest.fixture
def mock_trainer():
    """Provides a mock Lightning Trainer instance."""
    trainer = MagicMock(spec=L.Trainer)
    trainer.state = Mock()
    trainer.state.fn = "fit"
    trainer.current_epoch = 1
    trainer.global_step = 100
    return trainer


@pytest.fixture
def batch():
    """Provides a sample batch of data."""
    return {
        "input_ids": torch.randint(0, 1000, (4, 32)),
        "attention_mask": torch.ones(4, 32, dtype=torch.long),
    }


# --- Fixed Tests for GradNormMonitor ---


def test_grad_norm_monitor_basic(mock_pl_module, batch):
    callback = GradNormMonitor()
    mock_pl_module.log_dict = MagicMock()

    # Simulate backward pass
    optimizer = mock_pl_module.configure_optimizers()
    loss = mock_pl_module.training_step(batch, 0)
    loss.backward()

    callback.on_after_backward(trainer=None, pl_module=mock_pl_module)
    assert isinstance(callback.last_grad_norm, float)
    assert callback.last_grad_norm > 0

    callback.on_train_batch_end(
        trainer=None, pl_module=mock_pl_module, outputs=None, batch=batch, batch_idx=0
    )

    mock_pl_module.log_dict.assert_called_once()
    logged_metrics = mock_pl_module.log_dict.call_args[0][0]
    assert "gradient_norm" in logged_metrics
    assert "avg_query_length" in logged_metrics
    assert logged_metrics["avg_query_length"] == 32.0
    assert callback.last_grad_norm is None


def test_grad_norm_monitor_zero_gradients(mock_pl_module, batch):
    callback = GradNormMonitor()
    mock_pl_module.log_dict = MagicMock()

    # Zero out gradients
    for param in mock_pl_module.parameters():
        if param.requires_grad:
            param.grad = torch.zeros_like(param.data)

    callback.on_after_backward(trainer=None, pl_module=mock_pl_module)
    # Even with zero gradients, the norm should be 0.0, not None
    assert callback.last_grad_norm == 0.0


def test_grad_norm_monitor_no_gradients(mock_pl_module):
    callback = GradNormMonitor()
    mock_pl_module.log_dict = MagicMock()

    # No backward pass called, so no gradients
    callback.on_after_backward(trainer=None, pl_module=mock_pl_module)
    # When no gradients are present, clip_grad_norm_ returns 0.0
    assert callback.last_grad_norm == 0.0

    # Should not log anything if no gradients were computed
    callback.on_train_batch_end(
        trainer=None, pl_module=mock_pl_module, outputs=None, batch={}, batch_idx=0
    )
    mock_pl_module.log_dict.assert_not_called()


# --- Fixed Tests for PositionalEmbeddingTracker ---


def test_pe_tracker_initialization(mock_pl_module, mock_trainer):
    callback = PositionalEmbeddingTracker()
    mock_pl_module.log = MagicMock()

    # Correct path to positional embeddings
    pe_layer = mock_pl_module.lm.model.embeddings.position_embeddings
    initial_weights = pe_layer.weight.clone().detach()

    callback.on_train_start(mock_trainer, mock_pl_module)
    assert callback.initial_positional_embeddings is not None
    torch.testing.assert_close(callback.initial_positional_embeddings, initial_weights)
    assert (
        callback.initial_positional_embeddings.data_ptr() != pe_layer.weight.data_ptr()
    )


def test_pe_tracker_delta_calculation(mock_pl_module, mock_trainer):
    callback = PositionalEmbeddingTracker()
    mock_pl_module.log = MagicMock()

    callback.on_train_start(mock_trainer, mock_pl_module)

    # Modify weights
    pe_layer = mock_pl_module.lm.model.embeddings.position_embeddings
    with torch.no_grad():
        pe_layer.weight.data += 0.1

    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    mock_pl_module.log.assert_called_once()
    logged_delta = mock_pl_module.log.call_args[0][1]
    assert isinstance(logged_delta, torch.Tensor)
    assert logged_delta.item() > 0


def test_pe_tracker_no_initial_embeddings(mock_pl_module, mock_trainer):
    callback = PositionalEmbeddingTracker()
    mock_pl_module.log = MagicMock()

    # Don't call on_train_start, so initial_embeddings is None
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
    mock_pl_module.log.assert_not_called()


def test_pe_tracker_non_fit_state(mock_pl_module):
    callback = PositionalEmbeddingTracker()
    mock_pl_module.log = MagicMock()

    trainer = MagicMock()
    trainer.state.fn = "validate"  # Not in fit state

    callback.on_validation_epoch_end(trainer, mock_pl_module)
    mock_pl_module.log.assert_not_called()


# --- Fixed Tests for PositionalEmbeddingPCA ---


def test_pe_pca_computation():
    callback = PositionalEmbeddingPCA()

    # Create test data with known structure
    tensor = torch.randn(100, 10)
    tensor[:, 0] *= 10  # Make first dimension dominant

    pca_result = callback._perform_torch_pca(tensor, k=2)

    assert pca_result.shape == (100, 2)
    assert torch.var(pca_result[:, 0]) > torch.var(pca_result[:, 1])


def test_pe_pca_file_creation(mock_pl_module, mock_trainer):
    with tempfile.TemporaryDirectory() as tmp_dir:
        callback = PositionalEmbeddingPCA(output_dir=tmp_dir)

        # Create output directory as the callback would
        os.makedirs(tmp_dir, exist_ok=True)

        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

        # Check files were created
        epoch = mock_trainer.current_epoch
        step = mock_trainer.global_step

        csv_path = os.path.join(tmp_dir, f"pca_coords_epoch={epoch}_step={step}.csv")
        pdf_path = os.path.join(tmp_dir, f"pca_plot_epoch={epoch}_step={step}.pdf")

        assert os.path.exists(csv_path)
        assert os.path.exists(pdf_path)

        # Verify CSV content
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        max_pos = mock_pl_module.lm.model.config.max_position_embeddings
        assert data.shape == (max_pos, 2)


def test_pe_pca_non_fit_state(mock_pl_module):
    callback = PositionalEmbeddingPCA()

    trainer = MagicMock()
    trainer.state.fn = "validate"  # Not in fit state
    trainer.current_epoch = 1
    trainer.global_step = 100

    # Should not raise errors and not create files
    callback.on_validation_epoch_end(trainer, mock_pl_module)


def test_pe_pca_missing_embeddings(mock_pl_module, mock_trainer):
    callback = PositionalEmbeddingPCA()

    # Break the embedding access by patching the correct path
    with patch.object(mock_pl_module.lm.model.embeddings, "position_embeddings", None):
        # Should handle AttributeError gracefully
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
