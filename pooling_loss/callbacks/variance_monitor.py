# pooling_loss/callbacks/variance_monitor.py

import logging
from collections import deque
from typing import Any, Deque

import lightning as L
import torch


class VarianceMonitor(L.Callback):
    """A Lightning callback to monitor training stability by computing the
    variance of the loss over a rolling window and the variance of query
    lengths within a batch."""

    def __init__(self, window_size: int = 100):
        super().__init__()
        if window_size < 2:
            raise ValueError("window_size must be at least 2 to compute variance.")
        self.window_size = window_size
        self.losses: Deque[torch.Tensor] = deque(maxlen=self.window_size)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            if loss is None:
                logging.warning(
                    "VarianceMonitor failed to find 'loss' in training_step outputs."
                )
                return
        else:
            loss = outputs

        self.losses.append(loss.detach().cpu())

        if len(self.losses) > 1:
            loss_tensor = torch.stack(list(self.losses))
            rolling_variance = torch.var(loss_tensor)
            pl_module.log(
                "loss_variance_rolling",
                rolling_variance,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            query_lengths = attention_mask.sum(dim=1).float()
            length_variance = torch.var(query_lengths)
            pl_module.log(
                "query_length_variance_batch",
                length_variance,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
