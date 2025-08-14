# pooling_loss/utils/configs/data_config.py

from dataclasses import dataclass
from typing import Literal, Union

from pooling_loss.utils.helpers import DictAccessMixin


@dataclass
class DataConfig(DictAccessMixin):
    ds_name: Literal["allnli", "msmarco", "se"] = "allnli"
    batch_size: int = 32
    tokenizer_name: str = "FacebookAI/roberta-base"
    max_length: int = 512
    padding: Union[bool, str] = "max_length"  # max_length
    map_batch_size: int = 1000
    load_from_cache_file: bool = False
    shuffle: bool = False
