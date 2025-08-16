# pooling_loss/modules/dynamic_triplet_loss.py

from typing import TYPE_CHECKING, Dict

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pooling_loss.modules.base_loss import BaseLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class DynamicTripletLoss(BaseLoss):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__(cfg)
        assert (
            self.cfg.execution.margin is not None
        ), "Margin must be set in the configuration for DynamicTripletLoss"
        self.register_buffer("margin", torch.tensor(self.cfg.execution.margin))
        self.register_buffer("exp", torch.exp(torch.tensor(1)))

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
    ) -> Dict[str, torch.Tensor]:
        batch_size = query_embs.size(0)

        # Compute the (B, 2B) similarity matrix
        all_scores = self.pool(
            query_embs=query_embs,  # (B, S, H)
            key_embs=key_embs,  # (2B, S, H)
            q_mask=q_mask,  # (B, S)
            k_mask=k_mask,  # (2B, S)
        )

        all_dists = 1 - all_scores
        q_mask_sum = q_mask.sum(dim=1)

        targets = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[targets, targets]
        pos_dists = all_dists[targets, targets]
        negs = all_scores[targets, targets + batch_size]
        neg_dists = all_dists[targets, targets + batch_size]
        if self.cfg.execution.weighting == "log":
            dynamic_margin = self.margin * torch.log(self.exp + q_mask_sum.float() - 3)  # type: ignore
        else:
            dynamic_margin = self.margin * torch.sqrt(1 + q_mask_sum.float() - 3)  # type: ignore

        loss = F.relu(pos_dists - neg_dists + dynamic_margin).mean()

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
