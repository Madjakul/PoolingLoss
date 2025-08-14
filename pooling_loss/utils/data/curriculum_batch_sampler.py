# pooling_loss/utils/data/curriculum_batch_sampler.py


import random
from typing import Any


class CurriculumBatchSampler:
    def __init__(
        self, num_samples: int, batch_size: int, drop_last: bool = False
    ) -> None:
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Any:
        indices = list(range(self.num_samples))
        if self.drop_last:
            num_batches = self.num_samples // self.batch_size
            indices = indices[: num_batches * self.batch_size]
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i : i + self.batch_size]
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size
