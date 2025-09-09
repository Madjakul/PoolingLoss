# pooling_loss/utils/configs/model_config.py

from dataclasses import dataclass
from typing import Literal, Optional

from pooling_loss.utils.helpers import DictAccessMixin


@dataclass
class ModelConfig(DictAccessMixin):
    base_model_name: str = "FacebookAI/roberta-base"
    disable_pe: bool = False
    freeze_pe: bool = False  # only during training
    is_decoder_model: bool = False
    pooling_method: Literal["mean", "li", "average_li", "mean_li"] = "mean"
    chunk_size: Optional[int] = 64  # Only for LI methods
    q: Optional[float] = None  # Only if quantile_li
