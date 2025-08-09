# pooling_loss/modules/mteb_pooling_loss.py

import Lightning
import torch


class MTEBPoolingLoss(L.LightningModule):
    """MTEB Pooling Loss Module."""

    def __init__(self, model, loss_function, optimizer, scheduler=None):
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch["inputs"])
        loss = self.loss_function(outputs, batch["targets"])
        return loss

    def configure_optimizers(self):
        if self.scheduler:
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": self.scheduler,
            }
        return self.optimizer
