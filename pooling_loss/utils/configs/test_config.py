# pooling_loss/utils/configs/test_config.py

from dataclasses import dataclass
from typing import List, Literal

from pooling_loss.utils.helpers import DictAccessMixin


@dataclass
class TestConfig(DictAccessMixin):
    tasks: List[str] = []
    # --- trainer ---
    device: Literal["cpu", "gpu"] = "gpu"
    num_devices: int = 1
    process_group_backend: Literal["nccl", "gloo", "mpi"] = "gloo"
    # --- wandb ---
    use_wandb: bool = True
