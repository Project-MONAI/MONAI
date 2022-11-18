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

from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric


class LossMetric(CumulativeIterationMetric):
    """
    A wrapper to make ``loss_fn`` available as a cumulative metric. That is, the loss values computed from
    mini-batches can be combined in the ``reduction`` mode across multiple iterations, as a quantitative measurement
    of a model.

    Example:

    .. code-block:: python

        import torch
        from monai.losses import DiceLoss
        from monai.metrics import LossMetric

        dice_loss = DiceLoss(include_background=True)
        loss_metric = LossMetric(loss_fn=dice_loss)

        # first iteration
        y_pred = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])  # shape [batch=1, channel=1, 2, 2]
        y = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])  # shape [batch=1, channel=1, 2, 2]
        loss_metric(y_pred, y)

        # second iteration
        y_pred = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])  # shape [batch=1, channel=1, 2, 2]
        y = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])  # shape [batch=1, channel=1, 2, 2]
        loss_metric(y_pred, y)

        # aggregate
        print(loss_metric.aggregate(reduction="none"))  # tensor([[0.2000], [0.5000]]) (shape [batch=2, channel=1])

        # reset
        loss_metric.reset()
        print(loss_metric.aggregate())


    Args:
        loss_fn: a callable function that takes ``y_pred`` and optionally ``y`` as input (in the "batch-first" format),
            returns a "batch-first" tensor of loss values.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self, loss_fn: _Loss, reduction: Union[MetricReduction, str] = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
        """
        Returns the aggregated loss value across multiple iterations.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.
        """
        data = self.get_buffer()
        if data is None:
            return (torch.tensor(0.0), torch.tensor(0.0)) if self.get_not_nans else torch.tensor(0.0)
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor = None):  # type: ignore
        """
        Input `y_pred` is compared with ground truth `y`.
        Both `y_pred` and `y` are expected to be a batch-first Tensor (BC[HWD]).

        Returns:
             a tensor with shape (BC[HWD]), or a list of tensors, each tensor with shape (C[HWD]).
        """
        iter_loss = self.loss_fn(y_pred) if y is None else self.loss_fn(y_pred, y)
        if isinstance(iter_loss, torch.Tensor):
            while iter_loss.dim() < 2:
                iter_loss = iter_loss[None]
        # to be compatible with `Cumulative`, iter_loss should at least have a batch dim.
        # to be compatible with `do_metric_reduction`, iter_loss should at least have a batch and a channel dim.
        return iter_loss
