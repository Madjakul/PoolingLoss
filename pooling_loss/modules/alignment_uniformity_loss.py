# pooling_loss/modules/alignment_uniformity_loss.py

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int


class AlignmentUniformityLoss(nn.Module):

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "two_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "two_times_batch seq"],
    ) -> Dict[str, torch.Tensor]:
        batch_size = query_embs.size(0)

        # Get mean-pooled representations for each sequence
        # This gives us document-level embeddings to compare
        q_lengths = q_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, 1)
        k_lengths = k_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (2B, 1)

        # Mean pool, masking out padding tokens
        query_pooled = (query_embs * q_mask.unsqueeze(-1)).sum(
            dim=1
        ) / q_lengths  # (B, H)
        key_pooled = (key_embs * k_mask.unsqueeze(-1)).sum(dim=1) / k_lengths  # (2B, H)

        # L2 normalize the pooled representations
        query_pooled = F.normalize(query_pooled, p=2, dim=-1)
        key_pooled = F.normalize(key_pooled, p=2, dim=-1)

        # Extract positive pairs (first B keys correspond to positive examples)
        positive_keys = key_pooled[:batch_size]  # (B, H)
        # Compute squared L2 distance between positive pairs
        alignment_loss = (
            torch.norm(query_pooled - positive_keys, p=2, dim=-1).pow(2).mean()
        )

        # Combine all embeddings for uniformity calculation
        all_embeddings = torch.cat([query_pooled, key_pooled], dim=0)  # (3B, H)
        # Compute pairwise squared distances between all embeddings
        # This creates a (3B, 3B) matrix of squared L2 distances
        pairwise_dists = torch.cdist(all_embeddings, all_embeddings, p=2).pow(2)
        # Apply the uniformity formula: log E[exp(-2||f(x) - f(y)||²)]
        # We exclude diagonal (distance from embedding to itself = 0)
        mask = ~torch.eye(
            pairwise_dists.size(0), dtype=torch.bool, device=pairwise_dists.device
        )
        valid_dists = pairwise_dists[mask]
        log_uniformity_loss = torch.logsumexp(-2 * valid_dists, dim=0) - torch.log(
            torch.tensor(valid_dists.size(0), dtype=torch.float)
        )
        uniformity_loss = torch.exp(log_uniformity_loss)

        return {"alignment_loss": alignment_loss, "uniformity_loss": uniformity_loss}
