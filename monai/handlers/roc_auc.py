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

from typing import Callable, Union

from monai.handlers.ignite_metric import IgniteMetric
from monai.metrics import ROCAUCMetric
from monai.utils import Average


class ROCAUC(IgniteMetric):  # type: ignore[valid-type, misc]  # due to optional_import
    """
    Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC).
    accumulating predictions and the ground-truth during an epoch and applying `compute_roc_auc`.

    Args:
        average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
            Type of averaging performed if not binary classification. Defaults to ``"macro"``.

            - ``"macro"``: calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            - ``"weighted"``: calculate metrics for each label, and find their average,
                weighted by support (the number of true instances for each label).
            - ``"micro"``: calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            - ``"none"``: the scores for each class are returned.

        output_transform: a callable that is used to transform the
            :class:`~ignite.engine.Engine` `process_function` output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.

    Note:
        ROCAUC expects y to be comprised of 0's and 1's.
        y_pred must either be probability estimates or confidence values.

    """

    def __init__(
        self,
        average: Union[Average, str] = Average.MACRO,
        output_transform: Callable = lambda x: x,
    ) -> None:
        metric_fn = ROCAUCMetric(average=Average(average))
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            save_details=False,
        )
