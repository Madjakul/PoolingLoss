# pooling_loss/callbacks/grad_norm_monitor.py

import torch
from lightning.pytorch.callbacks import Callback


class GradNormMonitor(Callback):
    """A Lightning Callback to log the total gradient norm and average sequence
    length at each training step for later analysis."""

    def __init__(self):
        super().__init__()
        self.last_grad_norm = None

    def on_after_backward(self, trainer, pl_module):
        """After loss.backward(), compute the total gradient norm.

        This uses the fast, built-in PyTorch utility.
        """
        # torch.nn.utils.clip_grad_norm_ returns the total norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=float("inf")
        )
        # Temporarily store the computed norm
        self.last_grad_norm = total_norm.item()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """After the training step, log the gradient norm and the average
        sequence length."""
        if self.last_grad_norm is None:
            return

        # Calculate the average sequence length for the current batch
        query_lengths = batch["attention_mask"].sum(dim=1)
        avg_length = query_lengths.float().mean().item()

        # Log both metrics together for the same step
        # This ensures they are associated correctly in W&B or a CSV file
        metrics = {"gradient_norm": self.last_grad_norm, "avg_query_length": avg_length}
        pl_module.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=False)

        # Reset for the next step
        self.last_grad_norm = None
