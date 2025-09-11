# pooling_loss/utils/train_utils.py

import os.path as osp
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from pooling_loss.callbacks import (
    GradNormMonitor,
    PositionalEmbeddingPCA,
    PositionalEmbeddingTracker,
)
from pooling_loss.modules import PoolingLoss
from pooling_loss.utils.configs.base_config import BaseConfig
from pooling_loss.utils.data.allnli_datamodule import AllNLIDatamodule
from pooling_loss.utils.data.booksum_datamodule import BookSumDatamodule
from pooling_loss.utils.data.msmarco_datamodule import MSMarcoDatamodule
from pooling_loss.utils.data.unified_datamodule import UnifiedDatamodule


def setup_datamodule(
    cfg: BaseConfig,
    processed_ds_path: str,
    num_proc: int,
    individual_processed_paths: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> L.LightningDataModule:
    dm_map = {
        "allnli": AllNLIDatamodule,
        "booksum": BookSumDatamodule,
        "msmarco": MSMarcoDatamodule,
        "all": UnifiedDatamodule,
    }

    dm = dm_map[cfg.data.ds_name](
        cfg=cfg,
        processed_ds_path=processed_ds_path,
        num_proc=num_proc,
        cache_dir=cache_dir,
        individual_processed_paths=individual_processed_paths,
    )
    return dm


def setup_trainer(
    cfg: BaseConfig,
    model: torch.nn.Module,
    logs_dir: str,
    checkpoint_dir: Optional[str] = None,
) -> L.Trainer:
    # Set up callbacks
    callbacks = []

    name = (
        f"{cfg.model.base_model_name}-{cfg.data.ds_name}"
        f"-pooling:{cfg.model.pooling_method}-loss:{cfg.train.loss}"
    ).replace("/", "-")

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)
    callbacks.append(GradNormMonitor())
    callbacks.append(PositionalEmbeddingTracker())
    callbacks.append(PositionalEmbeddingPCA(output_dir=osp.join(logs_dir, name)))

    # Model checkpoint callback if checkpoint_dir is provided
    if checkpoint_dir is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath=osp.join(checkpoint_dir, name),
            filename="{epoch}",
            monitor=cfg.train.checkpoint_metric,
            mode=cfg.train.checkpoint_mode,
            save_top_k=cfg.train.save_top_k,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)

    # Configure loggers
    loggers = []
    if cfg.train.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            name=name,
            log_model=cfg.train.log_model,
            group=cfg.group_name,
            config=cfg.to_dict(),
        )
        wandb_logger.watch(
            model=model,
            log=cfg.train.watch,
            log_graph=False,
            log_freq=cfg.train.accumulate_grad_batches * 100,
        )
        loggers.append(wandb_logger)

    # Add CSV logger by default
    csv_logger = CSVLogger(save_dir=logs_dir, name=name)
    loggers.append(csv_logger)

    if cfg.train.strategy.startswith("ddp"):
        strategy = DDPStrategy(
            find_unused_parameters=cfg.train.strategy.endswith(
                "find_unused_parameters_true"
            ),
            process_group_backend=cfg.train.process_group_backend,
        )
    else:
        strategy = cfg.train.strategy

    trainer = L.Trainer(
        accelerator=cfg.train.device,
        strategy=strategy,
        devices=cfg.train.num_devices,
        max_steps=cfg.train.max_steps,
        max_epochs=cfg.train.max_epochs,
        val_check_interval=cfg.train.val_check_interval,
        enable_checkpointing=checkpoint_dir is not None,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=cfg.train.log_every_n_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        precision=cfg.train.precision,
        overfit_batches=cfg.train.overfit_batches,
    )
    return trainer


def train_tune(
    config: Dict[str, Any],
    processed_ds_path: str,
    individual_processed_paths: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
) -> None:
    cfg = BaseConfig(mode="tune").from_dict(config)

    dm = setup_datamodule(
        cfg,
        processed_ds_path=processed_ds_path,
        num_proc=cfg.tune.num_cpus_per_trial,
        individual_processed_paths=individual_processed_paths,
        cache_dir=cache_dir,
    )
    model = PoolingLoss(cfg)
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    callbacks.append(
        TuneReportCheckpointCallback(
            {
                "val_auroc": "val_auroc",
                "val_mrr": "val_mrr",
                "completed_epoch": "completed_epoch",
            },
            on="validation_epoch_end",
            save_checkpoints=False,
        )
    )
    loggers = []

    if cfg.tune.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.project_name,
            group=cfg.group_name,
            prefix="trial",
            log_model=False,
        )
        loggers.append(wandb_logger)

    trainer = L.Trainer(
        accelerator=cfg.tune.device,
        devices=cfg.tune.num_devices_per_trial,
        max_steps=cfg.tune.max_steps,
        max_epochs=cfg.tune.max_epochs,
        val_check_interval=cfg.tune.val_check_interval,
        callbacks=callbacks,
        enable_checkpointing=False,
        logger=loggers,
        log_every_n_steps=cfg.tune.log_every_n_steps,
        accumulate_grad_batches=cfg.tune.accumulate_grad_batches,
        gradient_clip_val=cfg.tune.gradient_clip_val,
        precision=cfg.tune.precision,
    )
    trainer.fit(model=model, datamodule=dm)
