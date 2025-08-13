# pooling_loss/utils/data/msmarco_datamodule.py

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


class MSMarcoDatamodule(L.LightningDataModule):
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
            batch["query"],
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
            "length": [sum(mask) for mask in tokenized_q["attention_mask"]],
        }

    def prepare_data(self) -> None:
        """Download the dataset if not already downloaded."""
        if (
            osp.exists(self.processed_ds_path)
            and osp.getsize(self.processed_ds_path) > 0
        ):
            return
        datasets.load_dataset("microsoft/ms_marco")

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
            path="microsoft/ms_marco",
            name="v1.1",
            cache_dir=self.cache_dir,
        )
        columns = ds["train"].column_names  # type: ignore

        logging.info("Filtering out train examples with no positive passages...")
        ds["train"] = ds["train"].filter(  # type: ignore
            self.filter_no_positive, num_proc=self.num_proc
        )
        logging.info("Creating train triplets from the dataset...")
        self.train_ds = ds["train"].map(  # type: ignore
            self.create_triplets,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=columns,
        )
        logging.info("Tokenizing train triplets...")
        self.train_ds = self.train_ds.map(
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=["positive", "negative", "query"],
        )
        logging.info("Filtering out validation examples with no positive passages...")
        ds["validation"] = ds["validation"].filter(  # type: ignore
            self.filter_no_positive, num_proc=self.num_proc
        )
        logging.info("Creating validation triplets from the dataset...")
        self.val_ds = ds["validation"].map(  # type: ignore
            self.create_triplets,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=columns,
        )
        logging.info("Tokenizing validation triplets...")
        self.val_ds = self.val_ds.map(
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=["positive", "negative", "query"],
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
            path="microsoft/ms_marco",
            name="v1.1",
            cache_dir=self.cache_dir,
        )
        columns = ds["test"].column_names  # type: ignore

        logging.info("Filtering out test examples with no positive passages...")
        ds["test"] = ds["test"].filter(  # type: ignore
            self.filter_no_positive, num_proc=self.num_proc
        )
        logging.info("Creating test triplets from the dataset...")
        self.test_ds = ds["test"].map(  # type: ignore
            self.create_triplets,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=columns,
        )
        logging.info("Tokenizing test triplets...")
        self.test_ds = self.test_ds.map(
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=["positive", "negative", "query"],
        )

        self.test_ds.set_format("torch")

        # --- Save the processed data to disk for future runs ---
        logging.info(f"Saving processed data to disk: {self.processed_ds_path}")
        os.makedirs(self.processed_ds_path, exist_ok=True)
        self.test_ds.save_to_disk(test_path)

    def train_dataloader(self) -> DataLoader:
        if self.cfg.mode == "tune":
            logging.info(f"Using a 1% subset of MSMarco for tuning.")
            num_samples = int(len(self.train_ds) * 0.01)
            train_ds = self.train_ds.select(range(num_samples))  # type: ignore
        else:
            train_ds = self.train_ds
        return DataLoader(
            train_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
            shuffle=True,
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

    @staticmethod
    def filter_no_positive(example: Dict[str, Any]) -> bool:
        """Filter out examples that have no positive passage."""
        return 1 in example["passages"]["is_selected"]

    @staticmethod
    def create_triplets(batch: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Map function to transform a batch of MS MARCO examples into query,
        positive, negative text triplets."""
        queries = []
        positives = []
        negatives = []

        for i in range(len(batch["query"])):
            query_text = batch["query"][i]
            passages = batch["passages"][i]

            try:
                positive_idx = passages["is_selected"].index(1)
                positive_text = passages["passage_text"][positive_idx]
            except ValueError:
                # Should not happen if we filter first, but as a safeguard
                continue

            # Use other passages as hard negatives
            for j, is_selected in enumerate(passages["is_selected"]):
                if is_selected:
                    continue
                negative_text = passages["passage_text"][j]
                queries.append(query_text)
                positives.append(positive_text)
                negatives.append(negative_text)

        return {"query": queries, "positive": positives, "negative": negatives}
