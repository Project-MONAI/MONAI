# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Union, Optional, Sequence

import torch

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from monai.utils.to_onehot import to_onehot

__all__ = [
    'MeanDice'
]


class MeanDice(Metric):
    """Computes dice score metric from full size Tensor and collects average.

    Args:
        remove_bg (Bool): skip dice computation on the first channel of the predicted output or not.
        logit_thresh (Float): the threshold value to round value to 0.0 and 1.0, default is 0.5.
        is_onehot_targets (Bool): whether the label data(y) is already in One-Hot format, will convert if not.
        output_transform (Callable): transform the ignite.engine.state.output into [y_pred, y] pair.
        device (torch.device): device specification in case of distributed computation usage.

    Note:
        This metric extends from Ignite Metric, for more details, please check:
        https://github.com/pytorch/ignite/tree/master/ignite/metrics

    """
    def __init__(
        self,
        remove_bg=True,
        logit_thresh=0.5,
        is_onehot_targets=False,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None
    ):
        super(MeanDice, self).__init__(output_transform, device=device)
        self.remove_bg = remove_bg
        self.logit_thresh = logit_thresh
        self.is_onehot_targets = is_onehot_targets

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]):
        assert len(output) == 2, 'MeanDice metric can only support y_pred and y.'
        y_pred, y = output

        average = self._function(y_pred, y)

        if len(average.shape) != 0:
            raise ValueError('_function did not return the average loss.')

        n = len(y)
        self._sum += average.item() * n
        self._num_examples += n

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'MeanDice must have at least one example before it can be computed.')
        return self._sum / self._num_examples

    def _function(self, y_pred, y):
        n_channels_y_pred = y_pred.shape[1]
        n_len = len(y_pred.shape)
        assert n_len == 4 or n_len == 5, 'unsupported input shape.'

        if self.is_onehot_targets is False:
            y = to_onehot(y, n_channels_y_pred)

        if self.remove_bg:
            y = y[:, 1:]
            y_pred = y_pred[:, 1:]

        y = (y >= self.logit_thresh).float()
        y_pred = (y_pred >= self.logit_thresh).float()

        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis = list(range(2, n_len))
        intersection = torch.sum(y * y_pred, reduce_axis)

        y_o = torch.sum(y, reduce_axis)
        y_pred_o = torch.sum(y_pred, reduce_axis)
        denominator = y_o + y_pred_o

        f = (2.0 * intersection) / denominator
        # final reduce_mean across batches and channels
        return torch.mean(f)
