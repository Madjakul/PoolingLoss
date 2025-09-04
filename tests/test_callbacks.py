# tests/test_callbacks.py

import os
import shutil
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

# --- Mocks and Fixtures ---


class MockLanguageModel(nn.Module):
    """A mock language model that mimics the structure of a Hugging Face
    model."""

    def __init__(self, config):
        super().__init__()
        # This nested structure mimics 'pl_module.lm.model'
        self.model = RobertaModel(config)

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
        # This structure is what the callbacks expect to find
        self.lm = MockLanguageModel(config)
        self.linear = nn.Linear(32, 2)  # Another trainable parameter

    def training_step(self, batch, batch_idx):
        # A dummy training step is needed for the trainer to run
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

    # --- FIX APPLIED HERE ---
    # Create a mock for the nested 'state' attribute
    trainer.state = Mock()
    trainer.state.fn = "fit"
    # -----------------------

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


# --- Tests for GradNormMonitor ---


def test_grad_norm_monitor(mock_pl_module, batch):
    """Tests the GradNormMonitor callback."""
    callback = GradNormMonitor()
    mock_pl_module.log_dict = MagicMock()

    # Simulate a backward pass
    optimizer = mock_pl_module.configure_optimizers()
    loss = mock_pl_module.training_step(batch, 0)
    loss.backward()

    # 1. Test on_after_backward
    callback.on_after_backward(trainer=None, pl_module=mock_pl_module)
    assert isinstance(callback.last_grad_norm, float)
    assert callback.last_grad_norm > 0

    # 2. Test on_train_batch_end
    callback.on_train_batch_end(
        trainer=None, pl_module=mock_pl_module, outputs=None, batch=batch, batch_idx=0
    )

    # Assert that logging was called correctly
    mock_pl_module.log_dict.assert_called_once()
    logged_metrics = mock_pl_module.log_dict.call_args[0][0]
    assert "gradient_norm" in logged_metrics
    assert "avg_query_length" in logged_metrics
    assert logged_metrics["avg_query_length"] == 32.0

    # Assert that the state is reset
    assert callback.last_grad_norm is None


# --- Tests for PositionalEmbeddingTracker ---


def test_pe_tracker_init_and_delta(mock_pl_module, mock_trainer):
    """Tests that PositionalEmbeddingTracker correctly tracks weight
    changes."""
    callback = PositionalEmbeddingTracker()
    # Correcting the typo from the original file
    callback.initial_positional_embeddings = None
    mock_pl_module.log = MagicMock()

    pe_layer = mock_pl_module.lm.model.roberta.embeddings.position_embeddings
    initial_weights = pe_layer.weight.clone().detach()

    # 1. Test on_train_start
    callback.on_train_start(mock_trainer, mock_pl_module)
    assert callback.initial_positional_embeddings is not None
    torch.testing.assert_close(callback.initial_positional_embeddings, initial_weights)
    # Ensure it's a copy, not a reference
    assert (
        callback.initial_positional_embeddings.data_ptr() != pe_layer.weight.data_ptr()
    )

    # Simulate a weight update
    with torch.no_grad():
        pe_layer.weight.data += 0.1

    # 2. Test on_validation_epoch_end
    callback.on_validation_epoch_end(mock_trainer, mock_pl_module)
    mock_pl_module.log.assert_called_once_with(
        "pos_emb_delta_l2", mock.ANY, on_step=False, on_epoch=True
    )

    # Check that the logged delta is a tensor and is greater than 0
    logged_delta = mock_pl_module.log.call_args[0][1]
    assert isinstance(logged_delta, torch.Tensor)
    assert logged_delta.item() > 0


# --- Tests for PositionalEmbeddingPCA ---


def test_pe_pca_torch_pca():
    """Tests the internal _perform_torch_pca method."""
    callback = PositionalEmbeddingPCA()
    # Create a tensor where variance is mostly along one dimension
    tensor = torch.randn(100, 10)
    tensor[:, 0] *= 10

    pca_result = callback._perform_torch_pca(tensor, k=2)
    assert pca_result.shape == (100, 2)
    # Variance should be higher in the first principal component
    assert torch.var(pca_result[:, 0]) > torch.var(pca_result[:, 1])


def test_pe_pca_file_creation(mock_pl_module, mock_trainer, tmp_path):
    """Tests that PositionalEmbeddingPCA creates the expected files."""
    output_dir = tmp_path / "pca_analysis"
    callback = PositionalEmbeddingPCA(output_dir=str(output_dir))

    # Correcting the path from the original file
    with patch(
        "pooling_loss.callbacks.pe_pca.PositionalEmbeddingPCA.on_validation_epoch_end",
        wraps=callback.on_validation_epoch_end,
    ):
        # Get the correct path by inspecting the mock module
        correct_path = "pooling_loss.callbacks.pe_pca.pl_module.lm.model.roberta.embeddings.position_embeddings.weight"

        # Run the callback hook
        callback.on_validation_epoch_end(mock_trainer, mock_pl_module)

    epoch = mock_trainer.current_epoch
    step = mock_trainer.global_step

    # Check for the data file (.csv)
    expected_csv_path = (
        output_dir / "data" / f"pca_coords_epoch={epoch}_step={step}.csv"
    )
    assert expected_csv_path.exists()

    # Check for the plot file (.pdf)
    expected_pdf_path = output_dir / "plots" / f"pca_plot_epoch={epoch}_step={step}.pdf"
    assert expected_pdf_path.exists()

    # Check content of CSV
    data = np.loadtxt(expected_csv_path, delimiter=",", skiprows=1)
    max_pos = mock_pl_module.lm.model.config.max_position_embeddings
    assert data.shape == (max_pos, 2)
