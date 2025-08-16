# pooling_loss/callbacks/latency_monitor.py

from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback


class LatencyMonitor(Callback):
    """A Lightning Callback to measure and log inference latency using
    torch.cuda.Event for higher precision and lower overhead."""

    def __init__(self, warmup_steps: int = 10):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.batch_times_ms = []
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def on_validation_batch_start(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record the start event of a validation batch."""
        if batch_idx < self.warmup_steps or not torch.cuda.is_available():
            return

        # Place a start marker in the CUDA stream
        self.start_event.record()

    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Record the end event and calculate latency."""
        if batch_idx < self.warmup_steps or not torch.cuda.is_available():
            return

        # Place an end marker in the CUDA stream & wait for it to complete
        self.end_event.record()
        torch.cuda.synchronize()  # Still need to wait for the event to finish

        # elapsed_time returns time in milliseconds
        batch_duration_ms = self.start_event.elapsed_time(self.end_event)

        try:
            # Adjust 'input_ids' to a key present in your batch
            batch_size = batch["input_ids"].size(0)
            if batch_size > 0:
                self.batch_times_ms.append(batch_duration_ms / batch_size)
        except (KeyError, AttributeError, ZeroDivisionError):
            print(
                "Warning: Could not determine batch size. Latency will be reported per batch."
            )
            self.batch_times_ms.append(batch_duration_ms)

    def on_validation_epoch_end(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> None:
        """Calculate and log the average latency for the epoch."""
        if not self.batch_times_ms:
            return

        avg_latency_ms = sum(self.batch_times_ms) / len(self.batch_times_ms)

        pl_module.log(
            "avg_latency_ms_per_query", avg_latency_ms, on_epoch=True, prog_bar=True
        )

        self.batch_times_ms.clear()
