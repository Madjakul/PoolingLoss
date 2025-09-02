# test.py

import json
import logging
import os.path as osp

import wandb
from mteb import MTEB

from pooling_loss.experiments.mteb_pooling_loss import MTEBPoolingLoss
from pooling_loss.utils.argparsers import TestArgparse
from pooling_loss.utils.configs import BaseConfig
from pooling_loss.utils.helpers import flatten_mteb_results
from pooling_loss.utils.logger import logging_config

logging_config()


if __name__ == "__main__":
    args = TestArgparse.parse_known_args()

    logging.info(f"Loading configuration from: {args.config_path}")
    cfg = BaseConfig.from_yaml(args.config_path)

    model_name = f"{cfg.model.base_model_name}".replace("/", "-")
    pe_status = "no-pe" if cfg.model.disable_pe else "pe"
    task_name_safe = "--".join(cfg.test.tasks).replace(",", "-").replace("/", "-")
    output_dir = osp.join(
        args.logs_dir,
        f"{cfg.group_name}",  # put loss, pooling method and max_length in group name
        f"{model_name}-{cfg.model.pooling_method}-{cfg.model.q}",
        cfg.data.ds_name,
        pe_status,
        task_name_safe,
    )

    # Initialize wandb run
    if cfg.test.use_wandb:
        wandb.init(
            project=cfg.project_name,
            group=cfg.group_name,
            name=str(output_dir).replace("/", "--"),
            config=cfg.to_dict(),
        )

    logging.info(f"Initializing model from checkpoint: {args.checkpoint_path}")
    model = MTEBPoolingLoss(cfg=cfg, checkpoint_path=args.checkpoint_path)

    logging.info(f"Selected local tasks: {cfg.test.tasks}")
    evaluation = MTEB(tasks=cfg.test.tasks)

    # Create a descriptive name for the output folder

    logging.info(f"Running MTEB evaluation. Results will be saved to: {output_dir}")

    # Run evaluation and get results
    results = evaluation.run(
        model,
        output_dir=str(output_dir),
        eval_splits=["test"],
    )

    # Log results to wandb
    flat_results = flatten_mteb_results(results)
    wandb.log(flat_results)

    # Also save results as a wandb table for better visualization
    results_table = wandb.Table(columns=["Task", "Metric", "Value"])
    for task_name, task_results in results.items():
        for split_name, split_results in task_results.items():
            for metric_name, metric_value in split_results.items():
                results_table.add_data(task_name, metric_name, metric_value)

    wandb.log({"MTEB Results": results_table})

    # Save results to a file and log as artifact
    results_file = osp.join(output_dir, "mteb_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    artifact = wandb.Artifact(
        name="mteb-results", type="evaluation", description="MTEB evaluation results"
    )
    artifact.add_file(results_file)
    wandb.log_artifact(artifact)

    logging.info("--- Evaluation complete ---")

    # Finish wandb run
    wandb.finish()
