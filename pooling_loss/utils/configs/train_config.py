# pooling_loss/utils/configs/train_config.py

from dataclasses import dataclass
from typing import Literal, Optional

from pooling_loss.utils.helpers import DictAccessMixin


@dataclass
class TrainConfig(DictAccessMixin):
    loss: Literal["info-nce", "triplet", "pairwise-ce"] = "info-nce"
    tau: Optional[float] = None  # Only for CE losses
    margin: Optional[float] = None  # Only for triplet loss
    # --- optimizer ---
    lr: float = 1e-5
    weight_decay: float = 0.01
    # --- checkpointing ---
    save_top_k: int = 2
    # --- trainer ---
    device: Literal["cpu", "gpu"] = "gpu"
    num_devices: int = 1
    strategy: str = "ddp_find_unused_parameters_true"
    process_group_backend: Literal["nccl", "gloo", "mpi"] = "gloo"
    max_steps: int = -1
    max_epochs: int = 2
    val_check_interval: Optional[float] = None
    check_val_every_n_epoch: Optional[int] = None
    log_every_n_steps: int = 10
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    precision: Literal["32", "16-mixed"] = "32"
    # --- wandb ---
    use_wandb: bool = True
    log_model: bool = True
    watch: Literal["gradients", "parameters", "all", "none"] = "gradients"
