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

from typing import Callable, Optional, Sequence, Union

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce

from monai.metrics.compute_meandice import compute_meandice

__all__ = ['MeanDice']


class MeanDice(Metric):
    """Computes dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(self,
                 include_background=True,
                 to_onehot_y=False,
                 mutually_exclusive=False,
                 add_sigmoid=False,
                 logit_thresh=0.5,
                 output_transform: Callable = lambda x: x,
                 device: Optional[Union[str, torch.device]] = None):
        """

        Args:
            include_background (Bool): whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y (Bool): whether to convert the output prediction into the one-hot format. Defaults to False.
            mutually_exclusive (Bool): if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False.
            add_sigmoid (Bool): whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            logit_thresh (Float): the threshold value to round value to 0.0 and 1.0. Defaults to None (no thresholding).
            output_transform (Callable): transform the ignite.engine.state.output into [y_pred, y] pair.
            device (torch.device): device specification in case of distributed computation usage.

        See also:
            monai.metrics.compute_meandice.compute_meandice
        """
        super(MeanDice, self).__init__(output_transform, device=device)
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.mutually_exclusive = mutually_exclusive
        self.add_sigmoid = add_sigmoid
        self.logit_thresh = logit_thresh

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
        scores = compute_meandice(y_pred, y, self.include_background, self.to_onehot_y, self.mutually_exclusive,
                                  self.add_sigmoid, self.logit_thresh)

        # add all items in current batch
        for batch in scores:
            not_nan = ~torch.isnan(batch)
            if not_nan.sum() == 0:
                continue
            class_avg = batch[not_nan].mean().item()
            self._sum += class_avg
            self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('MeanDice must have at least one example before it can be computed.')
        return self._sum / self._num_examples
