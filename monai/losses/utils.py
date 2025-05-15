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

from __future__ import annotations

import torch
import torch.linalg as LA


def compute_tp_fp_fn(
    input: torch.Tensor,
    target: torch.Tensor,
    reduce_axis: list[int],
    ord: int,
    soft_label: bool,
    decoupled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        input: the shape should be BNH[WD], where N is the number of classes.
        target: the shape should be BNH[WD] or B1H[WD], where N is the number of classes.
        reduce_axis: the axis to be reduced.
        ord: the order of the vector norm.
        soft_label: whether the target contains non-binary values (soft labels) or not.
            If True a soft label formulation of the loss will be used.
        decoupled: whether the input and the target should be decoupled when computing fp and fn.
            Only for the original implementation when soft_label is False.

    Adapted from:
        https://github.com/zifuwanggg/JDTLosses
    """

    # the original implementation that is erroneous with soft labels
    if ord == 1 and not soft_label:
        tp = torch.sum(input * target, dim=reduce_axis)
        # the original implementation of Dice and Jaccard loss
        if decoupled:
            fp = torch.sum(input, dim=reduce_axis) - tp
            fn = torch.sum(target, dim=reduce_axis) - tp
        # the original implementation of Tversky loss
        else:
            fp = torch.sum(input * (1 - target), dim=reduce_axis)
            fn = torch.sum((1 - input) * target, dim=reduce_axis)
    # the new implementation that is correct with soft labels
    # and it is identical to the original implementation with hard labels
    else:
        pred_o = LA.vector_norm(input, ord=ord, dim=reduce_axis)
        ground_o = LA.vector_norm(target, ord=ord, dim=reduce_axis)
        difference = LA.vector_norm(input - target, ord=ord, dim=reduce_axis)

        if ord > 1:
            pred_o = torch.pow(pred_o, exponent=ord)
            ground_o = torch.pow(ground_o, exponent=ord)
            difference = torch.pow(difference, exponent=ord)

        tp = (pred_o + ground_o - difference) / 2
        fp = pred_o - tp
        fn = ground_o - tp

    return tp, fp, fn
