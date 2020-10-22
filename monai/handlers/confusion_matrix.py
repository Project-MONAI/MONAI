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

from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.distributed as dist

from monai.metrics import compute_confusion_matrix
from monai.utils import Average, exact_version, optional_import

from .utils import all_gather

Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "reinit__is_reduced")


class ConfusionMatrix(Metric):  # type: ignore[valid-type, misc]  # due to optional_import
    """
    Compute confusion matrix related metrics. This function supports to calculate all metrics
    mentioned in: `Confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_.
    accumulating predictions and the ground-truth during an epoch and applying `compute_confusion_matrix`.

    Args:
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        activation: [``"sigmoid"``, ``"softmax"``]
            Activation method, if specified, an activation function will be employed for `y_pred`.
            Defaults to None.
            The parameter can also be a callable function, for example:
            ``activation = lambda x: torch.log_softmax(x)``.
        bin_mode: [``"threshold"``, ``"mutually_exclusive"``]
            Binarization method, if specified, a binarization manipulation will be employed
            for `y_pred`.

            - ``"threshold"``, a single threshold or a sequence of thresholds should be set.
            - ``"mutually_exclusive"``, `y_pred` will be converted by a combination of `argmax` and `to_onehot`.
        bin_threshold: the threshold for binarization, can be a single value or a sequence of
            values that each one of the value represents a threshold for a class.
        metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
            ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
            ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
            ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
            ``"informedness"``, ``"markedness"``]
            Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
            and you can also input those names instead.
        average: [``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``]
            Type of averaging performed if not binary classification.
            Defaults to ``"macro"``.

            - ``"macro"``: calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            - ``"weighted"``: calculate metrics for each label, and find their average,
                weighted by support (the number of true instances for each label).
            - ``"micro"``: calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            - ``"none"``: the scores for each class are returned.
        zero_division: the value to return when there is a zero division, for example, when all
            predictions and labels are negative. Defaults to 0.
        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.Engine` `process_function` output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
        device: device specification in case of distributed computation usage.

    """

    def __init__(
        self,
        to_onehot_y: bool = False,
        activation: Optional[Union[str, Callable]] = None,
        bin_mode: Optional[str] = "threshold",
        bin_threshold: Union[float, Sequence[float]] = 0.5,
        metric_name: str = "hit_rate",
        average: Union[Average, str] = Average.MACRO,
        zero_division: int = 0,
        output_transform: Callable = lambda x: x,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(output_transform, device=device)
        self.to_onehot_y = to_onehot_y
        self.activation = activation
        self.bin_mode = bin_mode
        self.bin_threshold = bin_threshold
        self.metric_name = metric_name
        self.average: Average = Average(average)
        self.zero_division = zero_division

    @reinit__is_reduced
    def reset(self) -> None:
        self._predictions: List[torch.Tensor] = []
        self._targets: List[torch.Tensor] = []

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        """
        Args:
            output: sequence with contents [y_pred, y].

        Raises:
            ValueError: When ``output`` length is not 2.
            ValueError: When ``y_pred`` dimension is not one of [1, 2].
            ValueError: When ``y`` dimension is not one of [1, 2].

        """
        if len(output) != 2:
            raise ValueError(f"output must have length 2, got {len(output)}.")
        y_pred, y = output
        if y_pred.ndimension() not in (1, 2):
            raise ValueError("Predictions should be of shape (batch_size, n_classes) or (batch_size, ).")
        if y.ndimension() not in (1, 2):
            raise ValueError("Targets should be of shape (batch_size, n_classes) or (batch_size, ).")

        self._predictions.append(y_pred.clone())
        self._targets.append(y.clone())

    def compute(self):
        _prediction_tensor = torch.cat(self._predictions, dim=0)
        _target_tensor = torch.cat(self._targets, dim=0)

        if dist.is_available() and dist.is_initialized() and not self._is_reduced:
            _prediction_tensor = all_gather(_prediction_tensor)
            _target_tensor = all_gather(_target_tensor)
            self._is_reduced = True

        return compute_confusion_matrix(
            y_pred=_prediction_tensor,
            y=_target_tensor,
            to_onehot_y=self.to_onehot_y,
            activation=self.activation,
            bin_mode=self.bin_mode,
            bin_threshold=self.bin_threshold,
            metric_name=self.metric_name,
            average=self.average,
            zero_division=self.zero_division,
        )
