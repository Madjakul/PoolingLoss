# pooling_loss/utils/data/allnli_datamodule.py

import logging
import os.path as osp
from typing import Any, Dict, List, Optional

import datasets
import lightning as L
from poolin_loss.utils.helpers import get_tokenizer
from torch.utils.data import DataLoader


class AllNLIDatamodule(L.LightningDataModule):
    def __init__(
        cfg: "BaseConfig",
        processed_ds_path: str,  # !!!!!!!!!!!!!!!!!!!!!!!
        tuning_mode: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.tuning_mode = tuning_mode
        self.processed_ds_path = processed_ds_path
        self.tokenizer = get_tokenizer(cfg.tokenizer_name)

    def prepare_data(self) -> None:
        """Download the dataset if not already downloaded."""
        if osp.exists(self.processed_ds_path):
            return
        datasets.load_dataset("sentence-transformers/all-nli")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            train_path = osp.join(self.processed_ds_path, "train")
            test_path = osp.join(self.processed_ds_path, "train")
        if stage == "test" or stage is None:
            pass
