# pooling_loss/utils/configs/test_config.py

from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pooling_loss.utils.helpers import DictAccessMixin


@dataclass
class TestConfig(DictAccessMixin):
    run_mteb: bool = False
    tasks: Optional[List[str]] = field(default_factory=list)
    # --- trainer ---
    gather: bool = False
    device: Literal["cpu", "gpu"] = "gpu"
    num_devices: int = 1
    process_group_backend: Literal["nccl", "gloo", "mpi"] = "gloo"
    # --- wandb ---
    use_wandb: bool = True
