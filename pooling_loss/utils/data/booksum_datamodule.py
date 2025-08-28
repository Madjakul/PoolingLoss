# pooling_loss/utils/data/booksum_datamodule.py

import logging
import os
import os.path as osp
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import datasets
import lightning as L
from torch.utils.data import DataLoader
from tqdm import tqdm

from pooling_loss.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from pooling_loss.utils.configs.base_config import BaseConfig


class BookSumDatamodule(L.LightningDataModule):
    def __init__(
        self,
        cfg: "BaseConfig",
        processed_ds_path: str,
        num_proc: int,
        cache_dir: Optional[str] = None,
        **kwargs,
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
            "length": [sum(mask) for mask in tokenized_q["attention_mask"]],  # type: ignore
        }

    def prepare_data(self) -> None:
        """Download the dataset if not already downloaded."""
        if (
            osp.exists(self.processed_ds_path)
            and osp.getsize(self.processed_ds_path) > 0
        ):
            return
        datasets.load_dataset("kmfoda/booksum")

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
            path="kmfoda/booksum",
            cache_dir=self.cache_dir,
        )
        logging.info("Filtering out train examples with no summary analysis...")
        ds["train"] = ds["train"].filter(  # type: ignore
            self.filter_summary_length, num_proc=self.num_proc
        )
        logging.info("Creating train triplets from the dataset...")
        self.train_ds = self.create_triplets(ds["train"])  # type: ignore
        logging.info("Tokenizing train triplets...")
        self.train_ds = self.train_ds.map(
            self.tokenize,
            batched=True,
            num_proc=self.num_proc,
            remove_columns=["positive", "negative", "query"],
        )
        logging.info("Filtering out validation examples with no summary analysis...")
        ds["validation"] = ds["validation"].filter(  # type: ignore
            self.filter_summary_length, num_proc=self.num_proc
        )
        logging.info("Creating validation triplets from the dataset...")
        self.val_ds = self.create_triplets(ds["validation"])  # type: ignore
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
            path="kmfoda/booksum",
            cache_dir=self.cache_dir,
        )
        logging.info("Filtering out test examples with no summary analysis...")
        ds["test"] = ds["test"].filter(  # type: ignore
            self.filter_summary_length, num_proc=self.num_proc
        )
        logging.info("Creating test triplets from the dataset...")
        self.test_ds = self.create_triplets(ds["test"])  # type: ignore
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
            logging.info(f"Using a 1% subset of Booksum for tuning.")
            num_samples = int(len(self.train_ds) * 0.005)
            head = self.train_ds.select(range(num_samples))  # type: ignore
            tail = self.train_ds.select(reversed(range(num_samples)))  # type: ignore
            train_ds = datasets.concatenate_datasets([head, tail]).sort("length")
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

    @staticmethod
    def filter_summary_length(example: Dict[str, Any]) -> bool:
        """Filter out examples that have no summary analysis."""
        return (
            example["analysis_length"] > 2
            # and example["analysis_length"] <= 510
            and example["summary_length"] > 2
            # and example["summary_length"] <= 510
        )

    @staticmethod
    def create_triplets(split_ds: datasets.Dataset) -> datasets.Dataset:
        """Transform Booksum examples into query, positive, negative text
        triplets."""
        queries = []
        positives = []
        negatives = []

        book_chapters = defaultdict(list)
        for example in tqdm(split_ds):
            book_chapters[example["bid"]].append(  # type: ignore
                {
                    "summary_text": example["summary_text"],  # type: ignore
                    "summary_analysis": example["summary_analysis"],  # type: ignore
                }
            )

        for _, chapters in tqdm(book_chapters.items()):
            if len(chapters) < 2:
                continue  # Need at least one negative
            for i, chap in enumerate(chapters):
                query_text = chap["summary_text"]
                positive_text = chap["summary_analysis"]
                for j in range(len(chapters)):
                    if i == j:
                        continue
                    negative_text = chapters[j]["summary_analysis"]
                    queries.append(query_text)
                    positives.append(positive_text)
                    negatives.append(negative_text)

        return datasets.Dataset.from_dict(
            {"query": queries, "positive": positives, "negative": negatives}
        )
