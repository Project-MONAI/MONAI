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

import inspect
import warnings
from typing import Callable, Optional, Union

import torch
from torch.nn.modules.loss import _Loss

__all__ = ["MaskedLoss"]


class MaskedLoss(_Loss):
    """
    This is a wrapper class for the loss functions.  It allows for additional
    weighting masks to be applied to both input and target.

    See Also:
        - :py:class:`monai.losses.MaskedDiceLoss`
    """

    def __init__(self, loss: Union[Callable, _Loss], *loss_args, **loss_kwargs) -> None:
        """
        Args:
            loss: loss function to be wrapped, this could be a loss class or an instance of a loss class.
            loss_args: arguments to the loss function's constructor if `loss` is a class.
            loss_kwargs: keyword arguments to the loss function's constructor if `loss` is a class.
        """
        super().__init__()
        self.loss = loss(*loss_args, **loss_kwargs) if inspect.isclass(loss) else loss
        if not callable(self.loss):
            raise ValueError("The loss function is not callable.")

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should be B1H[WD] or 11H[WD].
        """
        if mask is None:
            warnings.warn("No mask value specified for the MaskedLoss.")
            return self.loss(input, target)

        if input.dim() != mask.dim():
            warnings.warn(f"Dim of input ({input.shape}) is different from mask ({mask.shape}).")
        if input.shape[0] != mask.shape[0] and mask.shape[0] != 1:
            raise ValueError(f"Batch size of mask ({mask.shape}) must be one or equal to input ({input.shape}).")
        if target.dim() > 1:
            if mask.shape[1] != 1:
                raise ValueError(f"Mask ({mask.shape}) must have only one channel.")
            if input.shape[2:] != mask.shape[2:]:
                warnings.warn(f"Spatial size of input ({input.shape}) is different from mask ({mask.shape}).")
        return self.loss(input * mask, target * mask)
