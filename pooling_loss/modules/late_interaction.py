# pooling_loss/modules/late_interaction.py

import logging
import string
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class LateInteraction(torch.nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super().__init__()
        logging.info("Using Late Interaction pooling method")
        self.cfg = cfg

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model_name)
        punc_token_ids = set()
        for punc in string.punctuation:
            punc_token_ids.update(tokenizer.encode(punc, add_special_tokens=False))
        self.register_buffer(
            "punc_token_ids",
            torch.tensor(list(punc_token_ids), dtype=torch.long),
            persistent=False,
        )
        logging.info(
            f"Initialized Late Interaction with {len(self.punc_token_ids)} punctuation tokens to skip."
        )

    def forward(
        self,
        query_embs: Float[torch.Tensor, "batch seq hidden"],
        key_embs: Float[torch.Tensor, "n_times_batch seq hidden"],
        q_mask: Int[torch.Tensor, "batch seq"],
        k_mask: Int[torch.Tensor, "n_times_batch seq"],
        q_input_ids: Optional[Int[torch.Tensor, "batch seq"]] = None,
    ) -> Float[torch.Tensor, "batch n_times_batch"]:
        normalized_query_embs = F.normalize(query_embs, p=2, dim=-1)
        normalized_key_embs = F.normalize(key_embs, p=2, dim=-1)

        scores = torch.einsum(
            "ash, bth -> abst", normalized_query_embs, normalized_key_embs
        )
        # expanded_k_mask = k_mask.unsqueeze(0).unsqueeze(2).bool()
        # scores = scores.masked_fill(~expanded_k_mask, float("-inf"))
        scores = scores * k_mask.unsqueeze(0).unsqueeze(2)
        scores = scores.max(axis=-1).values

        if q_input_ids is not None and self.punc_token_ids.numel() > 0:
            punc_mask = torch.isin(q_input_ids, self.punc_token_ids)
            scores = scores.masked_fill(punc_mask.unsqueeze(1), 0.0)

        scores = scores * q_mask.unsqueeze(1).float()
        scores = scores.sum(axis=-1)
        return scores

        # scores = scores * q_mask.unsqueeze(1).unsqueeze(3)
        # scores = scores * k_mask.unsqueeze(0).unsqueeze(2)
        #
        # scores = scores.max(axis=-1).values.sum(axis=-1)
        # return scores
