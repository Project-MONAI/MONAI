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
from monai.metrics.compute_meandice import compute_meandice

__all__ = [
    'MeanDice'
]


class MeanDice(Metric):
    """Computes dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """
    def __init__(
        self,
        include_background=True,
        to_onehot_y=True,
        logit_thresh=None,
        add_sigmoid=False,
        mutually_exclusive=True,
        output_transform: Callable = lambda x: x,
        device: Optional[Union[str, torch.device]] = None
    ):
        """

        Args:
            include_background (Bool): whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y (Bool): whether to convert the output prediction into the one-hot format. Defaults to True.
            logit_thresh (Float): the threshold value to round value to 0.0 and 1.0. Defaults to None (no thresholding).
            add_sigmoid (Bool): whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            mutually_exclusive (Bool): if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to True.
            output_transform (Callable): transform the ignite.engine.state.output into [y_pred, y] pair.
            device (torch.device): device specification in case of distributed computation usage.

        See also:
            monai.metrics.compute_meandice.compute_meandice
        """
        super(MeanDice, self).__init__(output_transform, device=device)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.logit_thresh = logit_thresh
        self.add_sigmoid = add_sigmoid
        self.mutually_exclusive = mutually_exclusive

        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]):
        assert len(output) == 2, 'MeanDice metric can only support y_pred and y.'
        y_pred, y = output
        average = compute_meandice(y_pred, y, self.include_background, self.to_onehot_y, self.mutually_exclusive,
                                   self.add_sigmoid, self.logit_thresh)

        batch_size = len(y)
        self._sum += average.item() * batch_size
        self._num_examples += batch_size

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                'MeanDice must have at least one example before it can be computed.')
        return self._sum / self._num_examples
