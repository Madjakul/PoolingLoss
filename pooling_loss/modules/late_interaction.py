# pooling_loss/modules/chunked_late_interaction.py

# import logging
# from typing import TYPE_CHECKING
#
# import torch
# import torch.nn.functional as F
# from jaxtyping import Float, Int
#
# if TYPE_CHECKING:
#     from pooling_loss.utils.configs import BaseConfig
#
#
# class LateInteraction(torch.nn.Module):
#     """Implements a memory-efficient, key-chunked version of the standard late
#     interaction pooling."""
#
#     def __init__(self, cfg: "BaseConfig") -> None:
#         super().__init__()
#         logging.info("Using Key-Chunked 'Normal' Late Interaction")
#         # A single chunk size for iterating over keys
#         self.chunk_size = cfg.model.chunk_size
#
#     def forward(
#         self,
#         query_embs: Float[torch.Tensor, "batch seq hidden"],
#         key_embs: Float[torch.Tensor, "num_keys seq hidden"],
#         q_mask: Int[torch.Tensor, "batch seq"],
#         k_mask: Int[torch.Tensor, "num_keys seq"],
#     ) -> Float[torch.Tensor, "batch num_keys"]:
#
#         num_keys = key_embs.size(0)
#         all_chunk_scores = []
#
#         # 1. Prepare the FULL query tensor once, as it fits in memory
#         q_embs_norm = F.normalize(query_embs.unsqueeze(1), p=2, dim=-1)
#         q_mask_unsqueezed = q_mask.unsqueeze(1)
#
#         # 2. Loop ONLY over the keys in chunks
#         for i in range(0, num_keys, self.chunk_size):
#             # Get the current chunk of keys
#             key_chunk = key_embs[i : i + self.chunk_size]
#             k_mask_chunk = k_mask[i : i + self.chunk_size]
#
#             # Prepare key chunk tensors
#             k_embs_chunk_norm = F.normalize(key_chunk.unsqueeze(0), p=2, dim=-1)
#
#             # --- Score all queries against the current key chunk ---
#
#             # Compute similarity matrix between ALL queries and the KEY CHUNK
#             sim_matrix_chunk = torch.einsum(
#                 "insh, mjth->ijst", q_embs_norm, k_embs_chunk_norm
#             )
#             valid_mask_chunk = torch.einsum(
#                 "ixs, xjt->ijst",
#                 q_mask_unsqueezed.float(),
#                 k_mask_chunk.unsqueeze(0).float(),
#             ).bool()
#
#             # Apply your desired 0.0 masking
#             masked_sim_chunk = sim_matrix_chunk * valid_mask_chunk.float()
#
#             # Max-pool and sum to get the scores for this chunk
#             max_sim_values, _ = masked_sim_chunk.max(dim=-1)
#             chunk_scores = max_sim_values.sum(dim=-1)
#
#             all_chunk_scores.append(chunk_scores)
#
#         # 3. Concatenate results from all key chunks
#         scores = torch.cat(all_chunk_scores, dim=1)
#         return scores


# pooling_loss/modules/late_interaction.py

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class LateInteraction(torch.nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        logging.info("Using Late Interaction pooling method")
        self.cfg = cfg
        self.register_buffer("IGNORE", torch.tensor(float("-inf")))

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
    ) -> Float[torch.Tensor, "batch two_times_batch"]:
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

        # Max-based interaction
        masked_sim = sim_matrix.masked_fill(~valid_mask, self.IGNORE)  # type: ignore
        max_sim_values, _ = masked_sim.max(dim=-1)  # (B, B, S)
        is_padding_mask = q_mask == 0
        masked_max_sim = max_sim_values.masked_fill(is_padding_mask, 0.0)
        scores = masked_max_sim.sum(dim=-1)  # (B, 2B)
        return scores
