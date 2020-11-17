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

from typing import Callable, Optional, Sequence

import torch

from monai.metrics import ConfusionMatrixMetric
from monai.utils import exact_version, optional_import

NotComputableError, _ = optional_import("ignite.exceptions", "0.4.2", exact_version, "NotComputableError")
Metric, _ = optional_import("ignite.metrics", "0.4.2", exact_version, "Metric")
reinit__is_reduced, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "reinit__is_reduced")
sync_all_reduce, _ = optional_import("ignite.metrics.metric", "0.4.2", exact_version, "sync_all_reduce")


class ConfusionMatrix(Metric):  # type: ignore[valid-type, misc] # due to optional_import
    """
    Compute confusion matrix related metrics from full size Tensor and collects average over batch, class-channels, iterations.
    """

    def __init__(
        self,
        include_background: bool = True,
        metric_name: str = "hit_rate",
        compute_sample: bool = True,
        output_class: bool = False,
        output_transform: Callable = lambda x: x,
        device: Optional[torch.device] = None,
    ) -> None:
        """

        Args:
            include_background: whether to skip metric computation on the first channel of
                the predicted output. Defaults to True.
            metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
                ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
                ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
                ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
                ``"informedness"``, ``"markedness"``]
                Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
                and you can also input those names instead.
            compute_sample: if ``True``, each sample's metric will be computed first. Defaults to ``True``.
            output_class: if ``True``, scores for each class will be returned. The final result is each class's score.
                If average these scores, you will get the macro average of each class. Otherwise, the micro average score
                of each class will be returned. Defaults to ``False``.
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.

        See also:
            :py:meth:`monai.metrics.confusion_matrix`
        """
        super().__init__(output_transform, device=device)
        self.confusion_matrix = ConfusionMatrixMetric(
            include_background=include_background,
            metric_name=metric_name,
            compute_sample=compute_sample,
            output_class=output_class,
        )
        self._sum = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = 0.0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        """
        Args:
            output: sequence with contents [y_pred, y].

        Raises:
            ValueError: When ``output`` length is not 2. This metric can only support y_pred and y.

        """
        if len(output) != 2:
            raise ValueError(f"output must have length 2, got {len(output)}.")
        y_pred, y = output
        score, not_nans = self.confusion_matrix(y_pred, y)
        not_nans = int(not_nans.item())

        # add all items in current batch
        self._sum += score.item() * not_nans
        self._num_examples += not_nans

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self) -> float:
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        if self._num_examples == 0:
            raise NotComputableError("ConfusionMatrix metric must have at least one example before it can be computed.")
        return self._sum / self._num_examples
