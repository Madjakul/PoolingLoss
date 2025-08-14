# pooling_loss/modules/hard_margin_loss.py

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pooling_loss.modules.base_loss import BaseLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class HardMarginLoss(BaseLoss):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__(cfg)
        assert (
            cfg.execution.margin is not None
        ), "Margin must be set in the configuration for HardMarginLoss"

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

        targets = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[targets, targets]
        pos_dists = all_dists[targets, targets]
        negs = all_scores[targets, targets + batch_size]
        neg_dists = all_dists[targets, targets + batch_size]

        negative_pairs = neg_dists[
            neg_dists < (pos_dists.max() if len(pos_dists) > 1 else neg_dists.mean())
        ]
        positive_pairs = pos_dists[
            pos_dists > (neg_dists.min() if len(neg_dists) > 1 else pos_dists.mean())
        ]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.cfg.execution.margin - negative_pairs).pow(2).sum()  # type: ignore
        loss = (positive_loss + negative_loss) / batch_size

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
