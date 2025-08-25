# pooling_loss/modules/base_loss.py

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int

from pooling_loss.modules.dynamic_late_interaction import DynamicLateInteraction
from pooling_loss.modules.late_interaction import LateInteraction

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class BaseLoss(ABC, nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        self.cfg = cfg
        assert self.cfg.model.pooling_method in [
            "dli",
            "li",
            "mean",
        ], f"Invalid pooling method: {self.cfg.model.pooling_method}"
        if self.cfg.model.pooling_method == "dli":
            self.pool = DynamicLateInteraction(self.cfg)
        elif self.cfg.model.pooling_method == "li":
            self.pool = LateInteraction(self.cfg)
        else:
            self.pool = self.mean_pooling

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def mean_pooling(
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
        # Mean pooling and normalization
        query_vec = (query_embs * q_mask.unsqueeze(-1)).sum(dim=1)
        query_vec = F.normalize(query_vec, p=2, dim=-1)

        key_vec = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1)
        key_vec = F.normalize(key_vec, p=2, dim=-1)

        all_scores = torch.matmul(query_vec, key_vec.T)
        return all_scores
