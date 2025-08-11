# pooling_loss/utils/configs/model_config.py

from dataclasses import dataclass
from typing import Literal

from pooling_loss.utils.helpers import DictAccessMixin


@dataclass
class ModelConfig(DictAccessMixin):
    base_model_name: str = "FacebookAI/roberta-base"
    is_decoder_model: bool = False
    add_linear_layers: bool = True
    pooling_method: Literal["mean", "li", "dynamic_li"] = "mean"
