# pooling_loss/modules/pairwise_ce_loss.py

import logging
from typing import TYPE_CHECKING, Dict

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pooling_loss.modules.base_loss import BaseLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class PairwiseCELoss(BaseLoss):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__(cfg)
        assert (
            self.cfg.execution.tau is not None
        ), "Temperature must be set in the configuration for pariwise CE loss"
        logging.info(f"Using pariwise CE with tau={self.cfg.execution.tau}")
        self.register_buffer("tau", torch.tensor(self.cfg.execution.tau))

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "n_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "n_times_batch seq"],
        targets: Int[torch.Tensor, "n_times_batch"],
    ) -> Dict[str, torch.Tensor]:
        batch_size = query_embs.size(0)

        all_scores = self.pool(
            query_embs=query_embs,
            key_embs=key_embs,
            q_mask=q_mask,
            k_mask=k_mask,
        )
        all_scores = all_scores / self.tau

        local_targets = torch.arange(query_embs.size(0), device=query_embs.device)

        poss = all_scores[local_targets, targets]
        negs = all_scores[local_targets, targets + key_embs.size(0) // 2]

        logits = torch.stack([poss, negs], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query_embs.device)

        loss = F.cross_entropy(logits, labels, reduction="mean")

        return {
            "all_scores": all_scores,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
