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

from typing import Optional, Union

import torch
import warnings
import numpy as np

from monai.networks import one_hot
from monai.utils import Average


def _calculate(y, y_pred):
    assert y.ndimension() == y_pred.ndimension() == 1 and len(y) == len(
        y_pred
    ), "y and y_pred must be 1 dimension data with same length."
    assert y.unique().equal(
        torch.tensor([0, 1], dtype=y.dtype, device=y.device)
    ), "y values must be 0 or 1, can not be all 0 or all 1."
    n = len(y)
    indexes = y_pred.argsort()
    y = y[indexes].cpu().numpy()
    y_pred = y_pred[indexes].cpu().numpy()
    nneg = auc = tmp_pos = tmp_neg = 0

    for i in range(n):
        y_i = y[i]
        if i + 1 < n and y_pred[i] == y_pred[i + 1]:
            tmp_pos += y_i
            tmp_neg += 1 - y_i
            continue
        if tmp_pos + tmp_neg > 0:
            tmp_pos += y_i
            tmp_neg += 1 - y_i
            nneg += tmp_neg
            auc += tmp_pos * (nneg - tmp_neg / 2)
            tmp_pos = tmp_neg = 0
            continue
        if y_i == 1:
            auc += nneg
        else:
            nneg += 1
    return auc / (nneg * (n - nneg))


def compute_roc_auc(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    to_onehot_y: bool = False,
    softmax: bool = False,
    average: Union[Average, str] = Average.MACRO,
):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC). Referring to:
    `sklearn.metrics.roc_auc_score <https://scikit-learn.org/stable/modules/generated/
    sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_.

    Args:
        y_pred (torch.Tensor): input data to compute, typical classification model output.
            it must be One-Hot format and first dim is batch, example shape: [16] or [16, 2].
        y (torch.Tensor): ground truth to compute ROC AUC metric, the first dim is batch.
            example shape: [16, 1] will be converted into [16, 2] (where `2` is inferred from `y_pred`).
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        softmax: whether to add softmax function to `y_pred` before computation. Defaults to False.
        average: {``"macro"``, ``"weighted"``, ``"micro"``, ``"none"``}
            Type of averaging performed if not binary classification.
            Defaults to ``"macro"``.

            - ``"macro"``: calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            - ``"weighted"``: calculate metrics for each label, and find their average,
                weighted by support (the number of true instances for each label).
            - ``"micro"``: calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            - ``"none"``: the scores for each class are returned.

    Raises:
        ValueError: predictions should be of shape (batch_size, n_classes) or (batch_size, ).
        ValueError: targets should be of shape (batch_size, n_classes) or (batch_size, ).
        ValueError: unsupported average method.

    Note:
        ROCAUC expects y to be comprised of 0's and 1's. `y_pred` must be either prob. estimates or confidence values.

    """
    y_pred_ndim = y_pred.ndimension()
    y_ndim = y.ndimension()
    if y_pred_ndim not in (1, 2):
        raise ValueError("predictions should be of shape (batch_size, n_classes) or (batch_size, ).")
    if y_ndim not in (1, 2):
        raise ValueError("targets should be of shape (batch_size, n_classes) or (batch_size, ).")
    if y_pred_ndim == 2 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze(dim=-1)
        y_pred_ndim = 1
    if y_ndim == 2 and y.shape[1] == 1:
        y = y.squeeze(dim=-1)

    if y_pred_ndim == 1:
        if to_onehot_y:
            warnings.warn("y_pred has only one channel, to_onehot_y=True ignored.")
        if softmax:
            warnings.warn("y_pred has only one channel, softmax=True ignored.")
        return _calculate(y, y_pred)
    else:
        n_classes = y_pred.shape[1]
        if to_onehot_y:
            y = one_hot(y, n_classes)
        if softmax:
            y_pred = y_pred.float().softmax(dim=1)

        assert y.shape == y_pred.shape, "data shapes of y_pred and y do not match."

        average = Average(average)
        if average == Average.MICRO:
            return _calculate(y.flatten(), y_pred.flatten())
        else:
            y, y_pred = y.transpose(0, 1), y_pred.transpose(0, 1)
            auc_values = [_calculate(y_, y_pred_) for y_, y_pred_ in zip(y, y_pred)]
            if average == Average.NONE:
                return auc_values
            if average == Average.MACRO:
                return np.mean(auc_values)
            if average == Average.WEIGHTED:
                weights = [sum(y_) for y_ in y]
                return np.average(auc_values, weights=weights)
            raise ValueError("unsupported average method.")
