# pooling_loss/modules/dynamic_info_nce_loss.py

import torch.nn.functional as F

from pooling_loss.modules.base_loss import BaseLoss
from pooling_loss.modules.dynamic_late_interaction import DynamicLateInteraction
from pooling_loss.modules.late_interaction import LateInteraction


class DynamicInfoNCELoss(BaseLoss):

    def __init__(self) -> None:
        pass

    def forward(self):
        pass
