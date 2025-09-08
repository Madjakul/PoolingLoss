# pooling_loss/modules/dynamic_late_interaction.py

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class DynamicLateInteraction(torch.nn.Module):
    """Implements a late interaction mechanism where the final score is
    normalized by an "expected score".

    This expected score is derived from a statistical property (mean,
    median, or quantile) of the token-to-token similarity distribution.
    """

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("EPS", torch.tensor(1e-8))
        self.register_buffer("IGNORE", torch.tensor(float("-inf")))
        self.register_buffer("NAN", torch.tensor(float("nan")))

        if self.cfg.model.pooling_method == "quantile_li":
            assert (
                self.cfg.model.q is not None
            ), "A quantile value `q` must be set for the 'quantile_li' pooling method."
            self.register_buffer("q", torch.tensor(self.cfg.model.q))

    def _calculate_doc_stats(
        self,
        sim_matrix: Float[torch.Tensor, "batch two_batch seq seq"],
        valid_mask: Int[torch.Tensor, "batch two_batch seq seq"],
    ) -> Float[torch.Tensor, "batch two_batch"]:
        """Calculates a statistical measure over the token similarity matrix.

        This version is fully memory-efficient, using sampling with
        replacement for a consistent and robust approximation of
        median/quantile.
        """
        pooling_method = self.cfg.model.pooling_method

        if pooling_method == "mean_li":
            sum_sim = sim_matrix.masked_fill(~valid_mask.bool(), 0.0).sum(dim=(-2, -1))
            num_valid_pairs = valid_mask.sum(dim=(-2, -1)).float()
            return sum_sim / num_valid_pairs.clamp(min=self.EPS)
        else:
            num_samples = 2048
            batch_size, num_keys, _, _ = sim_matrix.shape

            # Reshape for batch processing
            sim_flat = sim_matrix.view(batch_size * num_keys, -1)
            mask_flat = valid_mask.view(batch_size * num_keys, -1)

            # Handle the unlikely edge case where a pair has no valid tokens
            if mask_flat.sum() == 0:
                return torch.zeros(batch_size, num_keys, device=sim_matrix.device)

            # Sample `num_samples` indices with replacement. This is efficient and
            # works even if num_samples > number of valid pairs.
            sampled_indices = torch.multinomial(
                mask_flat.float(), num_samples=num_samples, replacement=True
            )

            # Gather the scores using the sampled indices
            sampled_sims = torch.gather(sim_flat, -1, sampled_indices)
            sampled_sims = sampled_sims.view(batch_size, num_keys, num_samples)

            if pooling_method == "quantile_li":
                return torch.quantile(sampled_sims, self.q, dim=-1)
            else:  # median_li
                return torch.median(sampled_sims, dim=-1).values

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
        # 1. Prepare tensors for batched dot-product
        # (B, S, H) -> (B, 1, S, H)
        query_embs = F.normalize(query_embs.unsqueeze(1), p=2, dim=-1)
        # (2B, S, H) -> (1, 2B, S, H)
        key_embs = F.normalize(key_embs.unsqueeze(0), p=2, dim=-1)
        # (B, S) -> (B, 1, S)
        q_mask = q_mask.unsqueeze(1)
        # (2B, S) -> (1, 2B, S)
        k_mask = k_mask.unsqueeze(0)

        # 2. Compute token-level cosine similarities and valid interaction mask
        # -> (B, 2B, S, S)
        sim_matrix = torch.einsum("b n s h, m k t h -> b k s t", query_embs, key_embs)
        valid_mask = torch.einsum("b n s, m k t -> b k s t", q_mask, k_mask)

        # 3. Compute the statistical property (mean, median, etc.) for normalization
        doc_stats = self._calculate_doc_stats(sim_matrix, valid_mask)

        # 4. Calculate the expected score and detach it from the computation graph
        # We don't want to train the model to manipulate the score distribution.
        query_lengths = q_mask.sum(dim=-1).float()  # (B, 1)
        expected_scores = (doc_stats * query_lengths).detach()

        # 5. Compute the max-pooled similarity score (ColBERT-style)
        # Mask out invalid token pairs before the max operation
        masked_sim = sim_matrix.masked_fill(~valid_mask.bool(), self.IGNORE)
        # Max-pool over the key sequence dimension -> (B, 2B, S)
        max_sim_values, _ = masked_sim.max(dim=-1)
        # Mask out query padding tokens before the final sum
        max_sim_values = max_sim_values.masked_fill(~q_mask.bool(), 0.0)
        # Sum over the query sequence dimension -> (B, 2B)
        summed_max_sim = max_sim_values.sum(dim=-1)

        # 6. Normalize the score by the expected score
        scores = summed_max_sim / (expected_scores + self.EPS)

        return scores
