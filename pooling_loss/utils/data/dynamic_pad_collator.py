# pooling_loss/utils/data/dynamic_pad_collator.py


from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


class DynamicPadCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def _to_tensor(self, x):
        # preserves torch.Tensor without copying; converts lists/numpy if needed
        return torch.as_tensor(x, dtype=torch.long)

    def __call__(self, batch: List[Dict[str, Any]]):
        # assume batch is a list of dicts of tensors OR lists
        input_ids = [self._to_tensor(d["input_ids"]) for d in batch]
        attention_mask = [self._to_tensor(d["attention_mask"]) for d in batch]
        pos_input_ids = [self._to_tensor(d["pos_input_ids"]) for d in batch]
        pos_attention_mask = [self._to_tensor(d["pos_attention_mask"]) for d in batch]
        neg_input_ids = [self._to_tensor(d["neg_input_ids"]) for d in batch]
        neg_attention_mask = [self._to_tensor(d["neg_attention_mask"]) for d in batch]
        lengths = torch.as_tensor([int(d["length"]) for d in batch], dtype=torch.long)

        # pad_sequence accepts a list of 1D tensors and avoids extra copies where possible
        input_ids_p = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        attn_p = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        pos_input_ids_p = pad_sequence(
            pos_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        pos_attn_p = pad_sequence(pos_attention_mask, batch_first=True, padding_value=0)
        neg_input_ids_p = pad_sequence(
            neg_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        neg_attn_p = pad_sequence(neg_attention_mask, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids_p,
            "attention_mask": attn_p,
            "pos_input_ids": pos_input_ids_p,
            "pos_attention_mask": pos_attn_p,
            "neg_input_ids": neg_input_ids_p,
            "neg_attention_mask": neg_attn_p,
            "length": lengths,
        }
