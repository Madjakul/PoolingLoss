# pooling_loss/utils/configs/tune_config.py

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

from pooling_loss.utils.helpers import DictAccessMixin


@dataclass
class TuneConfig(DictAccessMixin):
    loss: Literal["info-nce", "triplet", "pairwise-ce"] = "info-nce"
    tau: Dict = field(default_factory=dict)
    margin: Optional[float] = None  # Only if margin-based lolss
    weighting: Optional[Literal["log", "sqrt", "none"]] = None
    # --- optimizer ---
    lr: Dict = field(default_factory=dict)
    weight_decay: Dict = field(default_factory=dict)
    # --- trainer ---
    device: Literal["cpu", "gpu"] = "gpu"
    num_devices_per_trial: int = 1
    num_cpus_per_trial: int = 10
    max_steps: int = -1
    max_epochs: int = 2
    log_every_n_steps: int = 10
    accumulate_grad_batches: int = 1
    gradient_clip_val: Optional[float] = None
    precision: Literal["32", "16-mixed"] = "32"
    # --- tuner ---
    metric: Literal["val_auroc", "val_mrr", "val_total_loss"] = "val_auroc"
    mode: Literal["min", "max"] = "max"
    num_samples: int = 30
    max_concurrent_trials: int = 3
    time_budget_s: int = 151200
    max_t: int = 2
    grace_period = 1
    val_check_interval: Optional[float] = None
    # --- wandb ---
    use_wandb: bool = True
    watch: str = "none"
