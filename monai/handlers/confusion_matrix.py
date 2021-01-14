# Copyright 2020 - 2021 MONAI Consortium
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

from monai.metrics import ConfusionMatrixMetric, compute_confusion_matrix_metric
from monai.metrics.utils import MetricReduction, do_metric_reduction
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
        compute_sample: bool = False,
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
            compute_sample: if ``True``, each sample's metric will be computed first.
                If ``False``, the confusion matrix for all samples will be accumulated first. Defaults to ``False``.
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
            reduction=MetricReduction.MEAN,
        )
        self._sum = 0.0
        self._num_examples = 0
        self.compute_sample = compute_sample
        self.metric_name = metric_name
        self._total_tp = 0.0
        self._total_fp = 0.0
        self._total_tn = 0.0
        self._total_fn = 0.0

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = 0.0
        self._num_examples = 0
        self._total_tp = 0.0
        self._total_fp = 0.0
        self._total_tn = 0.0
        self._total_fn = 0.0

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
        if self.compute_sample is True:
            score, not_nans = self.confusion_matrix(y_pred, y)
            not_nans = int(not_nans.item())

            # add all items in current batch
            self._sum += score.item() * not_nans
            self._num_examples += not_nans
        else:
            confusion_matrix = self.confusion_matrix(y_pred, y)
            confusion_matrix, _ = do_metric_reduction(confusion_matrix, MetricReduction.SUM)
            self._total_tp += confusion_matrix[0].item()
            self._total_fp += confusion_matrix[1].item()
            self._total_tn += confusion_matrix[2].item()
            self._total_fn += confusion_matrix[3].item()

    @sync_all_reduce("_sum", "_num_examples", "_total_tp", "_total_fp", "_total_tn", "_total_fn")
    def compute(self):
        """
        Raises:
            NotComputableError: When ``compute`` is called before an ``update`` occurs.

        """
        if self.compute_sample is True:
            if self._num_examples == 0:
                raise NotComputableError(
                    "ConfusionMatrix metric must have at least one example before it can be computed."
                )
            return self._sum / self._num_examples
        confusion_matrix = torch.tensor([self._total_tp, self._total_fp, self._total_tn, self._total_fn])
        return compute_confusion_matrix_metric(self.metric_name, confusion_matrix)
