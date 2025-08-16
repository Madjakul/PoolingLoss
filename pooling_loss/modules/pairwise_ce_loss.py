# pooling_loss/modules/pairwise_ce_loss.py

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pooling_loss.modules.base_loss import BaseLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class PairwiseCELoss(BaseLoss):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__(cfg)
        self.register_buffer("tau", torch.tensor(self.cfg.execution.tau))

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

        targets = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[targets, targets]
        negs = all_scores[targets, targets + batch_size]

        logits = torch.stack([poss, negs], dim=1) / self.tau  # type: ignore
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embs.device)

        loss = F.cross_entropy(logits, labels, reduction="mean")

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
