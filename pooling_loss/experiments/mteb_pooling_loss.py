# pooling_loss/experiments/mteb_pooling_loss.py

from typing import TYPE_CHECKING

import torch

from pooling_loss.modules import PoolingLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class MTEBPoolingLoss:
    """MTEB Pooling Loss Module."""

    def __init__(self, cfg: "BaseConfig", checkpoint_path: str) -> None:
        self.model = PoolingLoss(cfg)
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def encode(self):
        pass
