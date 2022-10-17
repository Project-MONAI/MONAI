# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from monai.utils import pytorch_after


class DeepSupervisionLoss(_Loss):
    """
    Wrapper class around the main loss function to accept a list of tensors returned from a deeply
    supervised networks. The final loss is computed as the sum of weighted losses for each of deep supervision levels.
    """

    def __init__(self, loss: _Loss, weight_mode: str = "exp", weights: Optional[List[float]] = None) -> None:
        """
        Args:
            loss: main loss instance, e.g DiceLoss().
            weight_mode: {``"same"``, ``"exp"``, ``"two"``}
                Specifies the weights calculation for each image level. Defaults to ``"exp"``.
                - ``"same"``: all weights are equal to 1.
                - ``"exp"``: exponentially decreasing weights by a power of 2: 0, 0.5, 0.25, 0.125, etc .
                - ``"two"``: equal smaller weights for lower levels: 1, 0.5, 0.5, 0.5, 0.5, etc
            weights: a list of weights to apply to each deeply supervised sub-loss, if provided, this will be used
                regardless of the weight_mode
        """
        super().__init__()
        self.loss = loss
        self.weight_mode = weight_mode
        self.weights = weights
        self.interp_mode = "nearest-exact" if pytorch_after(1, 11) else "nearest"

    def get_weight(self, level: int = 0) -> float:
        """
        Calculates a weight constant for a given image scale level
        """
        weight = 1.0
        if self.weights is not None and len(self.weights) > level:
            weight = self.weights[level]
        elif self.weight_mode == "same":
            weight = 1.0
        elif self.weight_mode == "exp":
            weight = max(0.5**level, 0.0625)
        elif self.weight_mode == "two":
            weight = 1.0 if level == 0 else 0.5

        return weight

    def get_loss(self, input: torch.Tensor, target: torch.Tensor):
        """
        Calculates a loss output accounting for differences in shapes,
        and downsizing targets if necessary (using nearest neighbor interpolation)
        Generally downsizing occurs for all level, except for the first (level==0)
        """
        if input.shape[2:] != target.shape[2:]:
            target = F.interpolate(target, size=input.shape[2:], mode=self.interp_mode)
        return self.loss(input, target)

    def forward(self, input: Union[torch.Tensor, List[torch.Tensor]], target: torch.Tensor):

        if isinstance(input, (list, tuple)):
            loss = torch.zeros(1, dtype=torch.float, device=target.device)
            for l in range(len(input)):
                loss += self.get_loss(input[l].float(), target) * self.get_weight(l)
            return loss
        return self.loss(input.float(), target)


ds_loss = DeepSupervisionLoss
