# train.py

import logging
import os
import random

import numpy as np
import torch

from pooling_loss.modules import PoolingLoss
from pooling_loss.utils import train_utils
from pooling_loss.utils.argparsers import TrainArgparse
from pooling_loss.utils.configs import BaseConfig
from pooling_loss.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging_config()
torch.cuda.empty_cache()


def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


if __name__ == "__main__":
    set_seed()
    args = TrainArgparse.parse_known_args()
    cfg = BaseConfig(mode="train").from_yaml(args.config_path)

    logging.info("--- Fine-tuning ---")
    model = PoolingLoss(cfg)

    trainer = train_utils.setup_trainer(
        cfg=cfg,
        model=model,
        logs_dir=args.logs_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    logging.info(
        f"Process {os.getpid()} -- "
        f"Rank: {trainer.global_rank}, "
        f"World Size: {trainer.world_size}"
    )

    logging.info("Preparing data module...")
    num_proc = int(args.num_proc / max(1, trainer.world_size))
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        processed_ds_path=args.processed_ds_path,
        num_proc=num_proc,
        individual_processed_paths=args.individual_processed_paths,
        cache_dir=args.cache_dir,
    )
    dm.setup("fit")

    trainer.fit(model=model, datamodule=dm)
    logging.info("--- Fine-tuning finished ---")
