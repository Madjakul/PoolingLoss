# pooling_loss/modules/dynamic_late_interaction.py

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class DynamicLateInteraction(torch.nn.Module):
    """Implements a late interaction mechanism where the final score is
    normalized by an "expected score".

    This expected score is derived from the mean of the token-level
    cosine similarities, effectively adjusting the max-pooled score by
    the average similarity across all valid token pairs.

    Notes
    -----
    This module uses sequential in-place operations on its internal similarity
    matrix to optimize memory usage. This is safe for backpropagation as the
    operations are on an intermediate tensor, but be aware that the internal
    `sim_matrix` tensor is modified during the forward pass.
    """

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        logging.info("Using Chunked Dynamic Late Interaction to save memory")
        self.cfg = cfg
        self.chunk_size = cfg.model.chunk_size
        self.register_buffer("EPS", torch.tensor(1e-8))
        self.register_buffer("IGNORE", torch.tensor(float("-inf")))

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "num_keys seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "num_keys seq"],
    ) -> Float[torch.Tensor, "batch num_keys"]:

        num_keys = key_embs.size(0)
        all_chunk_scores = []

        # Prepare query tensors once
        q_embs_norm = F.normalize(query_embs.unsqueeze(1), p=2, dim=-1)
        q_mask_unsqueezed = q_mask.unsqueeze(1)
        q_lengths = q_mask_unsqueezed.sum(dim=-1, keepdim=True).clamp(min=1).float()

        for i in range(0, num_keys, self.chunk_size):
            # 1. Get the current chunk of keys
            key_chunk = key_embs[i : i + self.chunk_size]
            k_mask_chunk = k_mask[i : i + self.chunk_size]

            # 2. Prepare key chunk tensors
            k_embs_chunk_norm = F.normalize(key_chunk.unsqueeze(0), p=2, dim=-1)
            k_mask_chunk_unsqueezed = k_mask_chunk.unsqueeze(0)

            # 3. Compute similarity matrix for the chunk (this is now small)
            sim_matrix_chunk = torch.einsum(
                "insh, mjth->ijst", q_embs_norm, k_embs_chunk_norm
            )
            valid_mask_chunk = torch.einsum(
                "ixs, xjt->ijst",
                q_mask_unsqueezed.float(),
                k_mask_chunk_unsqueezed.float(),
            ).bool()

            sim_matrix_chunk = sim_matrix_chunk.contiguous()
            sim_matrix_chunk.masked_fill_(~valid_mask_chunk, self.IGNORE)

            # 4. Compute max-pooled score for the chunk
            max_sim_values, _ = sim_matrix_chunk.max(dim=-1)
            max_sim_values.masked_fill_(~q_mask_unsqueezed.bool(), 0.0)
            summed_max_sim = max_sim_values.sum(dim=-1)

            # 5. Compute expected score for the chunk
            sim_sum = torch.where(valid_mask_chunk, sim_matrix_chunk, 0.0).sum(
                dim=(-1, -2)
            )
            valid_pair_count = valid_mask_chunk.sum(dim=(-1, -2))
            expected_scores = sim_sum / (valid_pair_count + self.EPS)
            expected_summed_scores = expected_scores * q_lengths.squeeze(-1)

            # 6. Compute final scores for this chunk and store them
            chunk_scores = summed_max_sim - expected_summed_scores
            all_chunk_scores.append(chunk_scores)

        # 7. Concatenate results from all chunks
        scores = torch.cat(all_chunk_scores, dim=1)
        return scores
