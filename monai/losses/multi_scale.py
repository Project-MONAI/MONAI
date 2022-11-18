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
from torch.nn.modules.loss import _Loss

from monai.networks.layers import gaussian_1d, separable_filtering
from monai.utils import LossReduction


def make_gaussian_kernel(sigma: int) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError(f"expecting positive sigma, got sigma={sigma}")
    return gaussian_1d(sigma=torch.tensor(sigma), truncated=3, approx="sampled", normalize=False)


def make_cauchy_kernel(sigma: int) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError(f"expecting positive sigma, got sigma={sigma}")
    tail = int(sigma * 5)
    k = torch.tensor([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
    k = torch.reciprocal(k)
    k = k / torch.sum(k)
    return k


kernel_fn_dict = {"gaussian": make_gaussian_kernel, "cauchy": make_cauchy_kernel}


class MultiScaleLoss(_Loss):
    """
    This is a wrapper class.
    It smooths the input and target at different scales before passing them into the wrapped loss function.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        loss: _Loss,
        scales: Optional[List] = None,
        kernel: str = "gaussian",
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
    ) -> None:
        """
        Args:
            loss: loss function to be wrapped
            scales: list of scalars or None, if None, do not apply any scaling.
            kernel: gaussian or cauchy.
        """
        super().__init__(reduction=LossReduction(reduction).value)
        if kernel not in kernel_fn_dict.keys():
            raise ValueError(f"got unsupported kernel type: {kernel}", "only support gaussian and cauchy")
        self.kernel_fn = kernel_fn_dict[kernel]
        self.loss = loss
        self.scales = scales

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if self.scales is None:
            loss: torch.Tensor = self.loss(y_pred, y_true)
        else:
            loss_list = []
            for s in self.scales:
                if s == 0:
                    # no smoothing
                    loss_list.append(self.loss(y_pred, y_true))
                else:
                    loss_list.append(
                        self.loss(
                            separable_filtering(y_pred, [self.kernel_fn(s).to(y_pred)] * (y_true.ndim - 2)),
                            separable_filtering(y_true, [self.kernel_fn(s).to(y_pred)] * (y_true.ndim - 2)),
                        )
                    )
            loss = torch.stack(loss_list, dim=0)

        if self.reduction == LossReduction.MEAN.value:
            loss = torch.mean(loss)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            loss = torch.sum(loss)  # sum over the batch and channel dims
        elif self.reduction != LossReduction.NONE.value:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return loss
