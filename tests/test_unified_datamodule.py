import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset
from torch.utils.data._utils.collate import default_collate

from pooling_loss.utils.configs.base_config import BaseConfig
from pooling_loss.utils.data import UnifiedDatamodule
from pooling_loss.utils.data.curriculum_batch_sampler import CurriculumBatchSampler
from pooling_loss.utils.data.dynamic_pad_collator import DynamicPadCollator


@pytest.fixture
def mock_config():
    config = MagicMock(spec=BaseConfig)
    config.data = MagicMock()
    config.data.tokenizer_name = "bert-base-uncased"
    config.data.batch_size = 4
    config.data.shuffle = False
    config.data.padding = False
    config.data.max_length = 512
    config.mode = "train"  # Not tune
    return config


@pytest.fixture
def small_dataset():
    # Create a small dataset with varying lengths
    data = {
        "input_ids": [[1] * i for i in range(1, 11)],
        "attention_mask": [[1] * i for i in range(1, 11)],
        "pos_input_ids": [[1] * i for i in range(1, 11)],
        "pos_attention_mask": [[1] * i for i in range(1, 11)],
        "neg_input_ids": [[1] * i for i in range(1, 11)],
        "neg_attention_mask": [[1] * i for i in range(1, 11)],
        "length": list(range(1, 11)),
    }
    ds = Dataset.from_dict(data)
    ds.set_format("torch")
    return ds.sort("length")


@pytest.fixture
def unified_datamodule(mock_config, small_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock individual paths, but since we override setup, not needed
        dm = UnifiedDatamodule(
            cfg=mock_config,
            individual_processed_paths=[tmpdir],
            unified_processed_path=os.path.join(tmpdir, "unified"),
            num_proc=1,
            cache_dir=None,
        )

        # Override setup to use small_dataset
        def mock_setup(stage=None):
            dm.train_ds = small_dataset
            dm.val_ds = small_dataset
            dm.test_ds = small_dataset

        dm.setup = mock_setup
        dm.setup()
        return dm


def test_curriculum_sampler(unified_datamodule):
    unified_datamodule.cfg.data.shuffle = False
    dataloader = unified_datamodule.train_dataloader()
    batch_indices = []
    batch_lengths = []

    for batch in dataloader:
        lengths = batch["length"].tolist()
        batch_lengths.append(lengths)
        # Since collator pads, but lengths are original
        # To get indices, but since sorted, check lengths increasing
    # Flatten lengths
    all_lengths = [l for b in batch_lengths for l in b]
    # Check overall sorted when sorted
    assert sorted(all_lengths) == list(
        range(1, 11)
    ), "Overall lengths do not match expected sorted lengths"
    # Check batch groups
    expected_groups = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]
    for i, bl in enumerate(batch_lengths):
        assert sorted(bl) == expected_groups[i], f"Batch {i} not grouped correctly"


def test_curriculum_sampler_separately(small_dataset):
    sampler = CurriculumBatchSampler(len(small_dataset), batch_size=4, drop_last=False)
    batches = list(sampler)
    assert len(batches) == 3  # 10/4 = 2 full + 1 partial
    # First batch: shuffled [0,1,2,3]
    # But since random, can't assert specific, but check len
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 2
    # Check that sorted(batch) == [0,1,2,3] etc.
    assert sorted(batches[0]) == [0, 1, 2, 3]
    assert sorted(batches[1]) == [4, 5, 6, 7]
    assert sorted(batches[2]) == [8, 9]


def test_random_shuffle(unified_datamodule):
    unified_datamodule.cfg.data.shuffle = True
    dataloader = unified_datamodule.train_dataloader()
    all_lengths = []
    for batch in dataloader:
        all_lengths.extend(batch["length"].tolist())
    # Since shuffled, sorted(all_lengths) should be 1-10, but all_lengths not sorted
    assert sorted(all_lengths) == list(range(1, 11))
    assert all_lengths != list(range(1, 11)), "Not shuffled"


def test_dynamic_collator(unified_datamodule, small_dataset):
    unified_datamodule.cfg.data.padding = False
    dataloader = unified_datamodule.train_dataloader()
    for batch in dataloader:
        # Check padding to max in batch
        max_len = max([len(ids) for ids in batch["input_ids"]])
        assert all(len(ids) == max_len for ids in batch["input_ids"])
        assert all(len(mask) == max_len for mask in batch["attention_mask"])
        # Check padding values
        for ids, mask in zip(batch["input_ids"], batch["attention_mask"]):
            pad_start = sum(mask)
            assert all(x == 1 for x in ids[:pad_start])  # Our dummy data
            assert all(x == 0 for x in ids[pad_start:])  # pad_token_id=0
        # Similarly for pos and neg


def test_pre_padded(unified_datamodule, small_dataset):
    # Modify small_dataset to be padded to max_length=10 for test
    max_l = 10
    padded_data = {
        "input_ids": [
            ids.tolist() + [0] * (max_l - len(ids))
            for ids in small_dataset["input_ids"]
        ],
        "attention_mask": [
            mask.tolist() + [0] * (max_l - len(mask))
            for mask in small_dataset["attention_mask"]
        ],
        "pos_input_ids": [
            ids.tolist() + [0] * (max_l - len(ids))
            for ids in small_dataset["pos_input_ids"]
        ],
        "pos_attention_mask": [
            mask.tolist() + [0] * (max_l - len(mask))
            for mask in small_dataset["pos_attention_mask"]
        ],
        "neg_input_ids": [
            ids.tolist() + [0] * (max_l - len(ids))
            for ids in small_dataset["neg_input_ids"]
        ],
        "neg_attention_mask": [
            mask.tolist() + [0] * (max_l - len(mask))
            for mask in small_dataset["neg_attention_mask"]
        ],
        "length": [l.item() for l in small_dataset["length"]],
    }
    padded_ds = Dataset.from_dict(padded_data)
    padded_ds.set_format("torch")
    unified_datamodule.train_ds = padded_ds
    unified_datamodule.cfg.data.padding = "max_length"  # Assume not False
    dataloader = unified_datamodule.train_dataloader()
    assert dataloader.collate_fn == default_collate  # Default collate
    for batch in dataloader:
        assert all(len(ids) == max_l for ids in batch["input_ids"])
        # Check attention_mask sums to length
        for mask, l in zip(batch["attention_mask"], batch["length"]):
            assert sum(mask) == l
