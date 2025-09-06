# pooling_loss/modules/language_model.py

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMaskedLM

if TYPE_CHECKING:
    from pooling_loss.utils.configs import BaseConfig


class LanguageModel(nn.Module):

    def __init__(self, cfg: "BaseConfig") -> None:
        super(LanguageModel, self).__init__()
        self.cfg = cfg
        self.disable_pe = cfg.model.disable_pe

        config = AutoConfig.from_pretrained(self.cfg.model.base_model_name)

        if self.cfg.model.disable_pe:
            logging.info("Disabling position embeddings.")

        if self.cfg.model.is_decoder_model:
            logging.info(
                f"Loading pretrained decoder from {self.cfg.model.base_model_name}."
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.base_model_name, config=config
            )
        else:
            logging.info(
                f"Loading pretrained encoder from {self.cfg.model.base_model_name}."
            )
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.cfg.model.base_model_name, config=config
            )

        if self.cfg.model.freeze_pe:
            self._freeze_pe()

        self.hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

    def _freeze_pe(self) -> None:
        # Freeze positional embeddings if specified
        if hasattr(self.model, "roberta") and hasattr(
            self.model.roberta.embeddings, "position_embeddings"
        ):
            self.model.roberta.embeddings.position_embeddings.weight.requires_grad = (
                False
            )
            logging.info("Froze RoBERTa positional embeddings.")
        else:
            logging.info(
                "No absolute positional embeddings to freeze (likely RoPE/relative)."
            )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch seq"],
        attention_mask: Int[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq hidden"]:
        if self.cfg.model.disable_pe:
            position_ids = torch.zeros_like(input_ids)
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        return out.hidden_states[-1]
