# test.py

import logging
from pathlib import Path

from mteb import MTEB

from pooling_loss.experiments.mteb_pooling_loss import MTEBPoolingLoss
from pooling_loss.utils.argparsers import TestArgparse
from pooling_loss.utils.configs import BaseConfig

if __name__ == "__main__":
    args = TestArgparse.parse_known_args()

    logging.info(f"Loading configuration from: {args.config_path}")
    cfg = BaseConfig.from_yaml(args.config_path)

    logging.info(f"Initializing model from checkpoint: {args.checkpoint_path}")
    model = MTEBPoolingLoss(cfg=cfg, checkpoint_path=args.checkpoint_path)

    tasks_to_run = [cfg.test.tasks]
    logging.info(f"Selected local tasks: {tasks_to_run}")
    evaluation = MTEB(tasks=tasks_to_run)

    # Create a descriptive name for the output folder
    model_name = Path(cfg.model.base_model_name).name
    pe_status = "no-pe" if cfg.model.disable_pe else "pe"
    output_folder = Path(args.output_dir) / f"{model_name}-{pe_status}-{task_name_safe}"

    logger.info(f"Running MTEB evaluation. Results will be saved to: {output_folder}")
    evaluation.run(
        model,
        output_folder=str(output_folder),
        eval_splits=["test"],  # Typically, we evaluate on the test split
    )
    logger.info("Evaluation complete.")
