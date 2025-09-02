# pooling_loss/callbacks/pe_tracker.py

import logging

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


class PositionalEmbeddingTracker(Callback):
    """Tracks the magnitude of change in positional embedding weights after
    each validation run."""

    def __init__(self):
        super().__init__()
        self.initial_position_al_embeddings = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Called when the train begins."""
        try:
            # For standard Hugging Face models like RoBERTa/BERT
            positional_embeddings = (
                pl_module.lm.model.roberta.embeddings.position_embeddings.weight
            )
            # Store a detached copy of the initial weights
            self.initial_positional_embeddings = positional_embeddings.clone().detach()
            logging.info(
                "Successfully stored initial positional embeddings for tracking."
            )
        except AttributeError:
            logging.error(
                "Could not find positional embeddings at the expected location."
            )
            self.initial_positional_embeddings = None

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Called when the validation epoch ends.

        This is the new hook for more frequent tracking.
        """
        if self.initial_positional_embeddings is not None and trainer.state.fn == "fit":
            try:
                # Get current weights
                current_positional_embeddings = (
                    pl_module.lm.model.roberta.embeddings.position_embeddings.weight
                )

                # Calculate the L2 norm of the difference
                delta = torch.norm(
                    current_positional_embeddings - self.initial_positional_embeddings,
                    p=2,
                )

                # Log the magnitude of the change
                pl_module.log("pos_emb_delta_l2", delta, on_step=False, on_epoch=True)
            except AttributeError:
                pass
