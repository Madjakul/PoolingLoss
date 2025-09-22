# test.py

import json
import logging
import os
import os.path as osp
import random

import numpy as np
import torch
from mteb import MTEB

import wandb
from pooling_loss.experiments.mteb_pooling_loss import MTEBPoolingLoss
from pooling_loss.modules import PoolingLoss
from pooling_loss.utils import train_utils
from pooling_loss.utils.argparsers import TestArgparse
from pooling_loss.utils.configs import BaseConfig
from pooling_loss.utils.helpers import flatten_mteb_results
from pooling_loss.utils.logger import logging_config

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging_config()
torch.cuda.empty_cache()


def set_seed(seed: int = 7):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logging.info(f"Random seed set as {seed}")


def run_standard_test(cfg: BaseConfig, args):
    """Runs the standard Lightning trainer.test() loop."""
    logging.info("--- Running Standard Test ---")
    logging.info(f"Loading model from checkpoint: {args.checkpoint_path}")

    # Load the model from the checkpoint
    model = PoolingLoss.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path, cfg=cfg
    )

    # Set up the logger
    logger = []
    if cfg.test.use_wandb:
        wandb_logger = wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=f"test-{cfg.data.ds_name}-{cfg.model.pooling_method}",
            config=cfg.to_dict(),
        )
        logger.append(wandb_logger)

    # Setup the trainer for testing
    trainer = train_utils.L.Trainer(
        accelerator=cfg.test.device,
        devices=cfg.test.num_devices,
        logger=logger,
        precision=cfg.train.precision if hasattr(cfg, "train") else "16-mixed",
    )

    logging.info("Preparing data module for testing...")
    dm = train_utils.setup_datamodule(
        cfg=cfg,
        processed_ds_path=args.processed_ds_path,
        num_proc=args.num_proc,  # Use a reasonable number of CPUs for data loading
        cache_dir=args.cache_dir,
    )

    # Run the test
    trainer.test(model=model, datamodule=dm)
    logging.info("--- Standard Test Finished ---")
    if cfg.test.use_wandb:
        wandb.finish()


def run_mteb_test(cfg: BaseConfig, args):
    assert cfg.test.tasks, "No MTEB tasks specified in the configuration."
    logging.info("--- Running MTEB Evaluation ---")
    model_name_safe = cfg.model.base_model_name.replace("/", "-")
    task_name_safe = "--".join(cfg.test.tasks).replace(",", "-").replace("/", "-")
    output_dir = osp.join(
        args.logs_dir,
        "mteb_results",
        f"{cfg.group_name}",
        f"{model_name_safe}-{cfg.model.pooling_method}",
        task_name_safe,
    )
    os.makedirs(output_dir, exist_ok=True)

    if cfg.test.use_wandb:
        wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=f"mteb-test-{cfg.model.pooling_method}",
            config=cfg.to_dict(),
        )

    logging.info(f"Initializing MTEB model from checkpoint: {args.checkpoint_path}")
    model = MTEBPoolingLoss(cfg=cfg, checkpoint_path=args.checkpoint_path)

    logging.info(f"Selected MTEB tasks: {cfg.test.tasks}")
    evaluation = MTEB(tasks=cfg.test.tasks)

    logging.info(f"Running MTEB evaluation. Results saved to: {output_dir}")
    results = evaluation.run(model, output_dir=str(output_dir), eval_splits=["test"])

    if cfg.test.use_wandb:
        flat_results = flatten_mteb_results(results)
        wandb.log(flat_results)
        # Add other wandb logging like tables and artifacts if needed
        wandb.finish()

    logging.info("--- MTEB Evaluation Complete ---")


if __name__ == "__main__":
    set_seed()
    # You might need to add/update the TestArgparse class to include all necessary args
    args = TestArgparse.parse_known_args()

    logging.info(f"Loading configuration from: {args.config_path}")
    cfg = BaseConfig.from_yaml(args.config_path)

    if not osp.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at path: {args.checkpoint_path}")

    # Decide which test to run based on the config flag
    if cfg.test.run_mteb:
        run_mteb_test(cfg, args)
    else:
        run_standard_test(cfg, args)
