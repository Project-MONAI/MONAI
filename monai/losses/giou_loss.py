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

from typing import Union

import torch
from torch.nn.modules.loss import _Loss

from monai.data.box_utils import COMPUTE_DTYPE, box_pair_giou
from monai.utils import LossReduction


class BoxGIoULoss(_Loss):

    """
    Compute the generalized intersection over union (GIoU) loss of a pair of boxes.
    The two inputs should have the same shape.

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``}
            Specifies the reduction to apply to the output. Defaults to ``"mean"``.
            - ``"none"``: no reduction will be applied.
            - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
            - ``"sum"``: the output will be summed.
    """

    def __init__(self, reduction: Union[LossReduction, str] = LossReduction.MEAN) -> None:
        super().__init__(reduction=LossReduction(reduction).value)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: predicted bounding boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``
            target: GT bounding boxes, Nx4 or Nx6 torch tensor. The box mode is assumed to be ``StandardMode``

        Raises:
            ValueError: When the two inputs have different shape.
        """
        if input.shape != target.shape:
            raise ValueError(
                f"input and target should have same shape. Got input.shape={input.shape}, "
                f"target.shape={target.shape}"
            )

        box_dtype = input.dtype
        giou: torch.Tensor = box_pair_giou(target.to(dtype=COMPUTE_DTYPE), input.to(dtype=COMPUTE_DTYPE))  # type: ignore
        loss: torch.Tensor = 1.0 - giou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        return loss.to(box_dtype)


giou = BoxGIoULoss
