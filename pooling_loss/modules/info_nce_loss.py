# pooling_loss/modules/info_nce_loss.py

from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pooling_loss.modules.base_loss import BaseLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class InfoNCELoss(BaseLoss):
    """InfoNCE Loss for deep stylometry models. This loss function computes the
    InfoNCE loss for a batch of query and key embeddings using in-batch
    negatives along with hard negatives.

    Parameters
    ----------
    cfg : BaseConfig
        Configuration object containing model and execution parameters, including the
        temperature parameter tau.

    Attributes
    ----------
    cfg : BaseConfig
        Configuration object with execution parameters.
    pool : nn.Module
        Pooling method used to compute similarity scores between query and key
        embeddings. Can be LateInteraction or mean pooling based on the configuration.
    tau : torch.Tensor
        Temperature parameter for scaling the similarity scores.
    """

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__(cfg)
        self.register_buffer("tau", torch.tensor(self.cfg.execution.tau))

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
        all_scaled_scores = all_scores / self.tau  # type: ignore

        targets = torch.arange(batch_size, device=query_embs.device)
        poss = all_scores[targets, targets]
        negs = all_scores[targets, targets + batch_size]

        loss = F.cross_entropy(all_scaled_scores, targets, reduction="mean")

        return {
            "all_scores": all_scores,
            "targets": targets,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
