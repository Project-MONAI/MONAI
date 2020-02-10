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

import torch
from typing import Callable, Union, Optional, Sequence
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from monai.utils.compute_meandice import compute_meandice

__all__ = [
    'MeanDice'
]


class MeanDice(Metric):
    """Computes dice score metric from full size Tensor and collects average.

    Args:
        remove_bg (Bool): skip dice computation on the first channel of the predicted output or not.
        is_onehot_targets (Bool): whether the label data(y) is already in One-Hot format, will convert if not.
        logit_thresh (Float): the threshold value to round value to 0.0 and 1.0, default is 0.5.
        output_transform (Callable): transform the ignite.engine.state.output into [y_pred, y] pair.
        device (torch.device): device specification in case of distributed computation usage.

    Note:
        (1) if this is multi-labels task(One-Hot label), use logit_thresh to convert y_pred to 0 or 1.
        (2) if this is multi-classes task(non-Ono-Hot label), use Argmax to select index and convert to One-Hot.
        This metric extends from Ignite Metric, for more details, please check:
        https://github.com/pytorch/ignite/tree/master/ignite/metrics

    """
    def __init__(
        self,
        remove_bg=True,
        is_onehot_targets=False,
        logit_thresh=0.5,
        add_sigmoid=False,
        add_softmax=False,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None
    ):
        super(MeanDice, self).__init__(output_transform, device=device)
        self.remove_bg = remove_bg
        self.is_onehot_targets = is_onehot_targets
        self.logit_thresh = logit_thresh
        self.add_sigmoid = add_sigmoid
        self.add_softmax = add_softmax

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]):
        assert len(output) == 2, 'MeanDice metric can only support y_pred and y.'
        y_pred, y = output

        average = compute_meandice(y_pred, y, self.remove_bg, self.is_onehot_targets,
                                   self.logit_thresh, self.add_sigmoid, self.add_softmax)

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
