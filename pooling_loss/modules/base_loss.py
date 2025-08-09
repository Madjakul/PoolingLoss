# pooling_loss/modules/base_loss.py

from abc import ABC, abstractmethod

import torch.nn as nn


class BaseLoss(ABC, nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def mean_pooling():
        pass
