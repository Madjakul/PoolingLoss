# pooling_loss/utils/data/unified_datamodule.py

import logging
import os
import os.path as osp
from typing import TYPE_CHECKING, List, Optional

import datasets
import lightning as L
from torch.utils.data import DataLoader

from pooling_loss.utils.data.curriculum_batch_sampler import CurriculumBatchSampler
from pooling_loss.utils.data.dynamic_pad_collator import DynamicPadCollator
from pooling_loss.utils.helpers import get_tokenizer

if TYPE_CHECKING:
    from pooling_loss.utils.configs.base_config import BaseConfig


class UnifiedDatamodule(L.LightningDataModule):
    def __init__(
        self,
        cfg: "BaseConfig",
        individual_processed_paths: List[str],
        processed_ds_path: str,
        num_proc: int,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.individual_paths = individual_processed_paths
        self.processed_ds_path = processed_ds_path
        self.num_proc = num_proc
        self.cache_dir = cache_dir
        self.tokenizer = get_tokenizer(cfg.data.tokenizer_name)

    def prepare_data(self) -> None:
        pass  # Individual datasets are already processed and tokenized on disk

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

        logging.info(
            "Processed unified data not found. Running concatenation and sorting..."
        )
        train_dss = []
        val_dss = []
        for ind_path in self.individual_paths:
            train_ind_path = osp.join(ind_path, "train")
            assert osp.exists(
                train_ind_path
            ), f"Train path {train_ind_path} does not exist."
            val_ind_path = osp.join(ind_path, "val")
            assert osp.exists(
                val_ind_path
            ), f"Validation path {val_ind_path} does not exist."
            train_dss.append(datasets.load_from_disk(train_ind_path))
            val_dss.append(datasets.load_from_disk(val_ind_path))

        logging.info(f"Concatenating and sorting datasets...")
        self.train_ds = datasets.concatenate_datasets(train_dss).sort("length")
        self.val_ds = datasets.concatenate_datasets(val_dss).sort("length")

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

        logging.info(
            "Processed unified data not found. Running concatenation and sorting..."
        )
        test_dss = []
        for ind_path in self.individual_paths:
            test_ind_path = osp.join(ind_path, "test")
            assert osp.exists(
                test_ind_path
            ), f"Test path {test_ind_path} does not exist."
            test_dss.append(datasets.load_from_disk(test_ind_path))

        self.test_ds = datasets.concatenate_datasets(test_dss).sort("length")

        self.test_ds.set_format("torch")

        # --- Save the processed data to disk for future runs ---
        logging.info(f"Saving processed data to disk: {self.processed_ds_path}")
        os.makedirs(self.processed_ds_path, exist_ok=True)
        self.test_ds.save_to_disk(test_path)

    def train_dataloader(self) -> DataLoader:
        if self.cfg.mode == "tune":
            logging.info(f"Using a 20% subset of Unified for tuning.")
            num_samples = int(len(self.train_ds) * 0.1)
            head = self.train_ds.select(range(num_samples))  # type: ignore
            tail = self.train_ds.select(reversed(range(num_samples)))  # type: ignore
            train_ds = datasets.concatenate_datasets([head, tail]).sort("length")
        else:
            train_ds = self.train_ds

        collator = None
        if self.cfg.data.padding is False:
            collator = DynamicPadCollator(pad_token_id=self.tokenizer.pad_token_id)

        if self.cfg.data.shuffle:
            # Shuffle everything randomly
            return DataLoader(
                train_ds,  # type: ignore
                batch_size=self.cfg.data.batch_size,
                num_workers=self.num_proc,
                shuffle=True,
                collate_fn=collator,
            )
        else:
            # Curriculum: sorted order, shuffle within batch
            batch_sampler = CurriculumBatchSampler(
                len(train_ds),  # type: ignore
                self.cfg.data.batch_size,
                drop_last=False,  # Can be configured if needed
            )
            return DataLoader(
                train_ds,  # type: ignore
                num_workers=self.num_proc,
                batch_sampler=batch_sampler,
                collate_fn=collator,
            )

    def val_dataloader(self) -> DataLoader:
        if self.cfg.mode == "tune":
            logging.info(f"Using a 40% subset of Unified for tuning validation.")
            num_samples = int(len(self.val_ds) * 0.2)
            head = self.val_ds.select(range(num_samples))  # type: ignore
            tail = self.val_ds.select(reversed(range(num_samples)))  # type: ignore
            val_ds = datasets.concatenate_datasets([head, tail]).sort("length")
        else:
            val_ds = self.val_ds

        collator = None
        if self.cfg.data.padding is False:
            collator = DynamicPadCollator(pad_token_id=self.tokenizer.pad_token_id)

        return DataLoader(
            val_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
            collate_fn=collator,
        )

    def test_dataloader(self) -> DataLoader:
        collator = None
        if self.cfg.data.padding is False:
            collator = DynamicPadCollator(pad_token_id=self.tokenizer.pad_token_id)

        return DataLoader(
            self.test_ds,  # type: ignore
            batch_size=self.cfg.data.batch_size,
            num_workers=self.num_proc,
            collate_fn=collator,
        )
