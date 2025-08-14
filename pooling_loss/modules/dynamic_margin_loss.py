# pooling_loss/modules/dynamic_margin_loss.py

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pooling_loss.modules.base_loss import BaseLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class DynamicMarginLoss(BaseLoss):
    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__(cfg)
        assert (
            self.cfg.execution.margin is not None
        ), "Margin must be set in the configuration for MarginLoss"

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
        gumbel_temp: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = query_embs.size(0)

        # Compute the (B, 2B) similarity matrix
        all_scores = self.pool(
            query_embs=query_embs,  # (B, S, H)
            key_embs=key_embs,  # (2B, S, H)
            q_mask=q_mask,  # (B, S)
            k_mask=k_mask,  # (2B, S)
            gumbel_temp=gumbel_temp,
        )
        all_dists = 1 - all_scores
        q_mask_sum = q_mask.sum(dim=1)

        targets = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[targets, targets]
        pos_dists = all_dists[targets, targets]
        negs = all_scores[targets, targets + batch_size]
        neg_dists = all_dists[targets, targets + batch_size]

        positive_loss = pos_dists.pow(2).sum()
        if self.cfg.execution.weighting == "log":
            dynamic_margin = self.cfg.execution.margin / torch.log(
                q_mask_sum.float() + 1
            )
        elif self.cfg.execution.weighting == "sqrt":
            dynamic_margin = self.cfg.execution.margin / torch.sqrt(
                q_mask_sum.float() + 1e-8
            )
        else:
            dynamic_margin = self.cfg.execution.margin / q_mask_sum.float()
        negative_loss = F.relu(dynamic_margin - neg_dists).pow(2).sum()
        loss = 0.5 * (positive_loss + negative_loss)

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
