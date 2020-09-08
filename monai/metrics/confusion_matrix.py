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

import numpy as np
import torch

from monai.metrics.confusion_matrix_utils import *
from monai.networks import one_hot
from monai.utils import Average


def compute_confusion_metric(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    to_onehot_y: bool = False,
    activation: Optional[Union[str, Callable]] = None,
    bin_mode: Optional[str] = "threshold",
    bin_threshold: Union[float, Sequence[float]] = 0.5,
    metric_name: str = "hit_rate",
    average: Union[Average, str] = Average.MACRO,
    zero_division: int = 0,
) -> Union[np.ndarray, List[float], float]:
    """
    Compute confusion matrix related metrics. This function supports to calculate all metrics
    mentioned in: `Confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_.
    Before calculating, an activation function and/or a binarization manipulation can be employed
    to pre-process the original inputs. Zero division is handled by replacing the result into a
    single value. Referring to:
    `sklearn.metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_.

    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [B] or [BN]. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        y: ground truth, the first dim is batch.
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
    Raises:
        AssertionError: when data shapes of `y_pred` and `y` do not match.
        AssertionError: when specify activation function and ``mutually_exclusive`` mode at the same time.
    """

    y_pred_ndim, y_ndim = y_pred.ndimension(), y.ndimension()
    # one-hot for ground truth
    if to_onehot_y:
        if y_pred_ndim == 1:
            warnings.warn("y_pred has only one channel, to_onehot_y=True ignored.")
        else:
            n_classes = y_pred.shape[1]
            y = one_hot(y, num_classes=n_classes)
    # check shape
    assert y.shape == y_pred.shape, "data shapes of y_pred and y do not match."
    # activation for predictions
    if activation is not None:
        assert bin_mode != "mutually_exclusive", "activation is unnecessary for mutually exclusive classes."
        y_pred = do_activation(y_pred, activation=activation)
    # binarization for predictions
    if bin_mode is not None:
        y_pred = do_binarization(y_pred, bin_mode=bin_mode, bin_threshold=bin_threshold)
    # get confusion matrix elements
    con_list = cal_confusion_matrix_elements(y_pred, y)
    # get simplified metric name
    metric_name = check_metric_name_and_unify(metric_name)
    result = do_calculate_metric(con_list, metric_name, average=average, zero_division=zero_division)
    return result
