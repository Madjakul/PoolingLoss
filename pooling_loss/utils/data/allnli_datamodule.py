# pooling_loss/utils/data/allnli_datamodule.py

import logging
import os
import os.path as osp
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import datasets
import lightning as L
from torch.utils.data import DataLoader

from pooling_loss.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from pooling_loss.utils.configs.base_config import BaseConfig


class AllNLIDatamodule(L.LightningDataModule):
    def __init__(
        self,
        cfg: "BaseConfig",
        processed_ds_path: str,
        num_proc: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_proc = num_proc
        self.processed_ds_path = processed_ds_path
        self.cache_dir = cache_dir
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    def tokenize(self, batch: Dict[str, List[str]]) -> Dict[str, Any]:
        tokenized_q = self.tokenizer(
            batch["anchor"],
            truncation=True,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
        )
        tokenized_pos = self.tokenizer(
            batch["positive"],
            truncation=True,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
        )
        tokenized_neg = self.tokenizer(
            batch["negative"],
            truncation=True,
            padding=self.cfg.data.padding,
            max_length=self.cfg.data.max_length,
        )
        return {
            "input_ids": tokenized_q["input_ids"],
            "attention_mask": tokenized_q["attention_mask"],
            "pos_input_ids": tokenized_pos["input_ids"],
            "pos_attention_mask": tokenized_pos["attention_mask"],
            "neg_input_ids": tokenized_neg["input_ids"],
            "neg_attention_mask": tokenized_neg["attention_mask"],
            "length": [sum(mask) for mask in tokenized_q["attention_mask"]],  # type: ignore
        }

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
            self.fit_setup()
        if stage == "test" or stage is None:
            self.test_setup()

    def fit_setup(self) -> None:
        train_path = osp.join(self.processed_ds_path, "train")
        val_path = osp.join(self.processed_ds_path, "val")

        if osp.exists(train_path) and osp.exists(val_path):
            logging.info(f"Loading processed data from disk: {self.processed_ds_path}")
            self.train_ds = datasets.load_from_disk(train_path)
            self.val_ds = datasets.load_from_disk(val_path)
            return

        logging.info("Processed data not found. Running full preprocessing pipeline...")
        ds = datasets.load_dataset(
            path="sentence-transformers/all-nli",
            name="triplet",
            cache_dir=self.cache_dir,
        )
        columns = ds["train"].column_names  # type: ignore

        self.train_ds = ds["train"].map(  # type: ignore
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=columns,
        )
        self.val_ds = ds["dev"].map(  # type: ignore
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=columns,
        )

        self.train_ds.set_format("torch")
        self.val_ds.set_format("torch")

        # --- Save the processed data to disk for future runs ---
        logging.info(f"Saving processed data to disk: {self.processed_ds_path}")
        os.makedirs(self.processed_ds_path, exist_ok=True)
        self.train_ds.save_to_disk(train_path)
        self.val_ds.save_to_disk(val_path)

    def test_setup(self) -> None:
        test_path = osp.join(self.processed_ds_path, "test")

        if osp.exists(test_path):
            logging.info(f"Loading processed data from disk: {self.processed_ds_path}")
            self.test_ds = datasets.load_from_disk(test_path)
            return

        logging.info("Processed data not found. Running full preprocessing pipeline...")
        ds = datasets.load_dataset(
            path="sentence-transformers/all-nli",
            name="triplet",
            cache_dir=self.cache_dir,
        )
        columns = ds["test"].column_names  # type: ignore

        self.test_ds = ds["test"].map(  # type: ignore
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=columns,
        )

        self.test_ds.set_format("torch")

        # --- Save the processed data to disk for future runs ---
        logging.info(f"Saving processed data to disk: {self.processed_ds_path}")
        os.makedirs(self.processed_ds_path, exist_ok=True)
        self.test_ds.save_to_disk(test_path)

    def train_dataloader(self) -> DataLoader:
        if self.cfg.mode == "tune":
            logging.info(f"Using a 2% subset of AllNLI for tuning.")
            num_samples = int(len(self.train_ds) * 0.02)
            train_ds = self.train_ds.select(range(num_samples))  # type: ignore
        else:
            train_ds = self.train_ds
        return DataLoader(
            train_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
            shuffle=self.cfg.data.shuffle,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
        )
