# pooling_loss/modules/late_interaction.py

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class DynamicLateInteraction(torch.nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("EPS", torch.tensor(1e-8))
        self.register_buffer("IGNORE", torch.tensor(float("-inf")))
        self.register_buffer("NAN", torch.tensor(float("nan")))
        if self.cfg.model.pooling_method == "quantile_li":
            assert (
                self.cfg.model.q is not None
            ), "Quantile value must be set for quantile_li pooling method"

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
        batch_size = query_embs.size(0)

        query_embs = query_embs.unsqueeze(1)  # (B, 1, S, H)
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        q_mask = q_mask.unsqueeze(1)  # (B, 1, S)
        key_embs = key_embs.unsqueeze(0)  # (1, 2B, S, H)
        key_embs = F.normalize(key_embs, p=2, dim=-1)
        k_mask = k_mask.unsqueeze(0)  # (1, 2B, S)

        # Compute token-level cosine similarities
        sim_matrix = torch.einsum("insh, mjth->ijst", query_embs, key_embs)

        # Compute valid mask for token pairs
        valid_mask = torch.einsum("ixs, xjt->ijst", q_mask, k_mask).bool()

        # Compute query lengths for scaling expected scores
        query_lengths = q_mask.sum(dim=-1).float()  # (B, 1)

        # Mask invalid similarities and flatten for percentile computation
        masked_sim = sim_matrix.masked_fill(~valid_mask, float("-inf"))
        flattened_sim = masked_sim.view(
            batch_size, -1, sim_matrix.size(-2) * sim_matrix.size(-1)
        )  # (B, 2B, S_q * S_k)
        # Replace -inf with NaN for percentile computation
        finite_mask = torch.isfinite(flattened_sim)
        flattened_sim = torch.where(finite_mask, flattened_sim, self.NAN)

        # Compute 75th percentile, ignoring -inf values
        if self.cfg.model.pooling_method == "quantile_li":
            doc_stats = torch.nanquantile(flattened_sim, self.cfg.model.q, dim=-1)
        elif self.cfg.model.pooling_method == "median_li":
            doc_stats = torch.nanmedian(flattened_sim, dim=-1).values
        else:
            doc_stats = torch.nanmean(flattened_sim, dim=-1)

        # Scale by query length to get expected scores
        # Make sure to detach so the model does not game the distributions
        expected_scores = (doc_stats * query_lengths) + self.EPS  # (B, 2B)
        expected_scores = expected_scores.detach()

        # Max-based interaction
        masked_sim = sim_matrix.masked_fill(~valid_mask, self.IGNORE)  # type: ignore
        max_sim_values, _ = masked_sim.max(dim=-1)  # (B, B, S)
        is_padding_mask = q_mask == 0
        masked_max_sim = max_sim_values.masked_fill(is_padding_mask, 0.0)
        scores = masked_max_sim.sum(dim=-1) / expected_scores  # (B, 2B)

        return scores
