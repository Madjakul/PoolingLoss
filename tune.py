# tune.py

import logging
import os

from pooling_loss.utils import tune_utils
from pooling_loss.utils.argparsers import TuneArgparse
from pooling_loss.utils.configs import BaseConfig
from pooling_loss.utils.logger import logging_config

os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["PYTHONUNBUFFERED"] = "1"


logging_config()


if __name__ == "__main__":
    args = TuneArgparse.parse_known_args()
    config = BaseConfig(mode="tune").from_yaml(args.config_path)
    logging.info(f"--- Tuning hyperparameters ---")
    logging.info(f"Config file: {args.config_path}")

    tuner = tune_utils.setup_tuner(
        config=config,
        ray_storage_path=args.ray_storage_path,
        processed_ds_path=args.processed_ds_path,
        individual_processed_paths=args.individual_processed_paths,
        cache_dir=args.cache_dir,
    )
    results = tuner.fit()
    logging.info("--- Tuning finished ---")
