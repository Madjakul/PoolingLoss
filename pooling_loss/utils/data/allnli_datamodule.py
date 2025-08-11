# pooling_loss/utils/data/allnli_datamodule.py

import logging
import os.path as osp
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import datasets
import lightning as L
from torch.utils.data import DataLoader

from pooling_loss.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from pooling_loss.utils.configs.base_config import BaseConfig


class AllNLIDatamodule(L.LightningDataModule):
    def __init__(self, cfg: "BaseConfig", processed_ds_path: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.processed_ds_path = processed_ds_path
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    def prepare_data(self) -> None:
        """Download the dataset if not already downloaded."""
        if (
            osp.exists(self.processed_ds_path)
            and osp.getsize(self.processed_ds_path) > 0
        ):
            return
        datasets.load_dataset("sentence-transformers/all-nli")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_path = osp.join(self.processed_ds_path, "train")
            test_path = osp.join(self.processed_ds_path, "train")
        if stage == "test" or stage is None:
            pass

    def train_dataloader(self) -> DataLoader:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass
