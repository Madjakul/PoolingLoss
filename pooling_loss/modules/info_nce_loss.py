# pooling_loss/modules/info_nce_loss.py

import logging
from typing import TYPE_CHECKING, Dict

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

from pooling_loss.modules.base_loss import BaseLoss

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class InfoNCELoss(BaseLoss):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__(cfg)
        assert (
            self.cfg.execution.tau is not None
        ), "Temperature must be set in the configuration for infoNCE loss"
        logging.info(f"Using InfoNCE Loss with tau={self.cfg.execution.tau}")
        self.register_buffer("tau", torch.tensor(self.cfg.execution.tau))

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "n_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "n_times_batch seq"],
        targets: Int[torch.Tensor, "n_times_batch"],
        q_input_ids: Int[torch.Tensor, "batch seq"],
    ) -> Dict[str, torch.Tensor]:
        # Split keys and masks into positive and negative sets
        num_pos_keys = key_embs.size(0) // 2
        pos_key_embs = key_embs[:num_pos_keys]
        neg_key_embs = key_embs[num_pos_keys:]

        pos_k_mask = k_mask[:num_pos_keys]
        neg_k_mask = k_mask[num_pos_keys:]

        # Score queries against positive keys
        pos_scores = self.pool(
            query_embs=query_embs,
            key_embs=pos_key_embs,
            q_mask=q_mask,
            k_mask=pos_k_mask,
            q_input_ids=q_input_ids,
        )

        # Score queries against negative keys
        neg_scores = self.pool(
            query_embs=query_embs,
            key_embs=neg_key_embs,
            q_mask=q_mask,
            k_mask=neg_k_mask,
            q_input_ids=q_input_ids,
        )

        # Concatenate scores for the final loss calculation
        all_scores = torch.cat([pos_scores, neg_scores], dim=1)

        all_scaled_scores = all_scores / self.tau  # type: ignore
        local_targets = torch.arange(query_embs.size(0), device=query_embs.device)

        # Positive scores are on the diagonal of the first concatenated block
        poss = all_scores[local_targets, targets]
        # Negative scores are on the diagonal of the second concatenated block
        negs = all_scores[local_targets, targets + num_pos_keys]

        loss = F.cross_entropy(all_scaled_scores, targets, reduction="mean")

        return {
            "all_scores": all_scores,
            "poss": poss,
            "negs": negs,
            "loss": loss,
        }
