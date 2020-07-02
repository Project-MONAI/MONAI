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

from monai.metrics import DiceMetric
from monai.utils import exact_version, optional_import, MetricReduction

NotComputableError, _ = optional_import("ignite.exceptions", "0.3.0", exact_version, "NotComputableError")
Metric, _ = optional_import("ignite.metrics", "0.3.0", exact_version, "Metric")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.3.0", exact_version, "reinit__is_reduced")
sync_all_reduce, _ = optional_import("ignite.metrics.metric", "0.3.0", exact_version, "sync_all_reduce")


class MeanDice(Metric):
    """
    Computes Dice score metric from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        mutually_exclusive: bool = False,
        sigmoid: bool = False,
        logit_thresh: float = 0.5,
        output_transform: Callable = lambda x: x,
        device: Optional[torch.device] = None,
    ):
        """

        Args:
            include_background: whether to include dice computation on the first channel of the predicted output.
                Defaults to True.
            to_onehot_y: whether to convert the output prediction into the one-hot format. Defaults to False.
            mutually_exclusive: if True, the output prediction will be converted into a binary matrix using
                a combination of argmax and to_onehot. Defaults to False.
            sigmoid: whether to add sigmoid function to the output prediction before computing Dice.
                Defaults to False.
            logit_thresh: the threshold value to round value to 0.0 and 1.0. Defaults to None (no thresholding).
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device (torch.device): device specification in case of distributed computation usage.

        See also:
            :py:meth:`monai.metrics.meandice.compute_meandice`
        """
        super().__init__(output_transform, device=device)
        self.dice = DiceMetric(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            mutually_exclusive=mutually_exclusive,
            sigmoid=sigmoid,
            logit_thresh=logit_thresh,
            reduction=MetricReduction.MEAN,
        )
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, dict]]):
        if not len(output) == 2:
            raise ValueError("MeanDice metric can only support y_pred and y.")
        y_pred, y = output
        score = self.dice(y_pred, y)
        not_nans = self.dice.not_nans.item()

        # add all items in current batch
        self._sum += score.item() * not_nans
        self._num_examples += not_nans

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("MeanDice must have at least one example before it can be computed.")
        return self._sum / self._num_examples
