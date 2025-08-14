# pooling_loss/utils/data/dynamic_pad_collator.py

from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


class DynamicPadCollator:
    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [torch.tensor(d["input_ids"]) for d in batch],  # type: ignore
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(d["attention_mask"]) for d in batch],  # type: ignore
            batch_first=True,
            padding_value=0,
        )
        pos_input_ids = pad_sequence(
            [torch.tensor(d["pos_input_ids"]) for d in batch],  # type: ignore
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        pos_attention_mask = pad_sequence(
            [torch.tensor(d["pos_attention_mask"]) for d in batch],  # type: ignore
            batch_first=True,
            padding_value=0,
        )
        neg_input_ids = pad_sequence(
            [torch.tensor(d["neg_input_ids"]) for d in batch],  # type: ignore
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        neg_attention_mask = pad_sequence(
            [torch.tensor(d["neg_attention_mask"]) for d in batch],  # type: ignore
            batch_first=True,
            padding_value=0,
        )
        lengths = torch.tensor([d["length"] for d in batch])  # type: ignore

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "neg_input_ids": neg_input_ids,
            "neg_attention_mask": neg_attention_mask,
            "length": lengths,
        }
