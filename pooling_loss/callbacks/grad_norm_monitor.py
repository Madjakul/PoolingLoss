# pooling_loss/callbacks/grad_norm_monitor.py

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


class GradNormMonitor(Callback):
    """A Lightning Callback to log the total gradient norm at each optimizer
    step for later analysis."""

    def __init__(self):
        super().__init__()
        self.grad_norm_dict = {}

    @rank_zero_only
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        """After gradients have been accumulated and unscaled, but before the
        optimizer step, compute the total gradient norm."""
        # The `torch.nn.utils.clip_grad_norm_` function returns the total norm
        # of all parameters in `pl_module.parameters()` before clipping.
        # We set max_norm to infinity to only compute the norm without clipping.
        total_norm = torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(), max_norm=float("inf")
        )
        self.grad_norm_dict["gradient_norm"] = total_norm.item()

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """After the training step, log the gradient norm and the average
        sequence length if an optimizer step occurred."""
        # This hook runs after every batch, but the optimizer step only
        # happens every `accumulate_grad_batches`. We log only when a norm
        # has been computed in `on_before_optimizer_step`.
        if not self.grad_norm_dict or not batch:
            return

        pl_module.log_dict(
            self.grad_norm_dict,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.grad_norm_dict.clear()
