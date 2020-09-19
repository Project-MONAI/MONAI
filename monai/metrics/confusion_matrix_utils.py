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

import warnings
from typing import Callable, List, Sequence, Union

import numpy as np
import torch

from monai.networks import one_hot
from monai.utils import Average


def do_activation(input_data: torch.Tensor, activation: Union[str, Callable] = "softmax") -> torch.Tensor:
    """
    This function is used to do activation for inputs.

    Args:
        input_data: the input that to be activated, in the shape [B] or [BN] or [BNHW] or [BNHWD].
        activation: can be ``"sigmoid"`` or ``"softmax"``, or a callable function. Defaults to ``"softmax"``.
            An example for callable function: ``activation = lambda x: torch.log_softmax(x)``.

    Raises:
        NotImplementedError: When input an activation name that is not implemented.
    """
    input_ndim = input_data.ndimension()
    if activation == "softmax":
        if input_ndim == 1:
            warnings.warn("input_data has only one channel, softmax ignored.")
        else:
            input_data = input_data.float().softmax(dim=1)
    elif activation == "sigmoid":
        input_data = input_data.float().sigmoid()
    elif callable(activation):
        input_data = activation(input_data)
    else:
        raise NotImplementedError("activation can only be sigmoid, softmax or a callable function.")
    return input_data


def do_binarization(
    input_data: torch.Tensor,
    bin_mode: str = "threshold",
    bin_threshold: Union[float, Sequence[float]] = 0.5,
) -> torch.Tensor:
    """
    Args:
        input_data: the input that to be binarized, in the shape [B] or [BN] or [BNHW] or [BNHWD].
        bin_mode: can be ``"threshold"`` or ``"mutually_exclusive"``, or a callable function.
            - ``"threshold"``, a single threshold or a sequence of thresholds should be set.
            - ``"mutually_exclusive"``, `input_data` will be converted by a combination of
            argmax and to_onehot.
        bin_threshold: the threshold to binarize the input data, can be a single value or a sequence of
            values that each one of the value represents a threshold for a class.

    Raises:
        AssertionError: when `bin_threshold` is a sequence and the input has the shape [B].
        AssertionError: when `bin_threshold` is a sequence but the length != the number of classes.
        AssertionError: when `bin_mode` is ``"mutually_exclusive"`` the input has the shape [B].
        AssertionError: when `bin_mode` is ``"mutually_exclusive"`` the input has the shape [B, 1].
    """
    input_ndim = input_data.ndimension()
    if bin_mode == "threshold":
        if isinstance(bin_threshold, Sequence):
            assert input_ndim > 1, "a sequence of thresholds are used for multi-class tasks."
            error_hint = "the length of the sequence should be the same as the number of classes."
            assert input_data.shape[1] == len(bin_threshold), "{}".format(error_hint)
            for cls_num in range(input_data.shape[1]):
                input_data[:, cls_num] = (input_data[:, cls_num] > bin_threshold[cls_num]).float()
        else:
            input_data = (input_data > bin_threshold).float()
    elif bin_mode == "mutually_exclusive":
        assert input_ndim > 1, "mutually_exclusive is used for multi-class tasks."
        n_classes = input_data.shape[1]
        assert n_classes > 1, "mutually_exclusive is used for multi-class tasks."
        input_data = torch.argmax(input_data, dim=1, keepdim=True)
        input_data = one_hot(input_data, num_classes=n_classes)
    return input_data


def cal_confusion_matrix_elements(p: torch.Tensor, t: torch.Tensor) -> List[np.ndarray]:
    """
    This function is used to calculate the number of true positives (tp), true negatives(tn),
    false positives (fp), false negatives (fn), total positives and total negatives, and
    return a list of these values.

    Args:
        p: predictions, a binarized torch.Tensor that its first dimension represents the batch size.
        t: ground truth, a binarized torch.Tensor that its first dimension represents the batch size.
            parameter t and p should have same shapes.

    Notes:
        If the input shape is [B], each element in the returned list is an int value.
        Else, each element in the returned list is an np.ndarray with shape (N,), where each element in
        this array represents the value for the corresponding class.

    Raises:
        AssertionError: when `p` and `t` have different shapes.
    """
    assert p.shape == t.shape, "predictions and targets should have same shapes."
    with torch.no_grad():
        dims = p.ndimension()
        if dims > 1:  # in the form of [BNS], where S is the number of pixels for one sample.
            batch_size, n_class = p.shape[:2]
            p = p.view(batch_size, n_class, -1)
            t = t.view(batch_size, n_class, -1)
        tp = ((p + t) == 2).float()
        tn = ((p + t) == 0).float()
        if dims > 1:
            tp = tp.sum(dim=[0, 2])
            tn = tn.sum(dim=[0, 2])
            total_p = t.sum(dim=[0, 2])
            total_n = batch_size * t.shape[-1] - total_p
        else:
            tp, tn = tp.sum(), tn.sum()
            total_p = t.sum()
            total_n = t.shape[-1] - total_p
        fn = total_p - tp
        fp = total_n - tn
        result = [tp, tn, fp, fn, total_p, total_n]
        result = [l.data.cpu().numpy() for l in result]
    return result


def handle_zero_divide(
    numerator: Union[np.ndarray, torch.Tensor, float, int],
    denominator: Union[np.ndarray, torch.Tensor, float, int],
    zero_division: int = 0,
) -> Union[np.ndarray, torch.Tensor, float]:
    """
    This function is used to handle the division case that the denominator has 0.
    This function takes sklearn for reference, see:
    https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/metrics/_classification.py#L1179
    """
    if isinstance(denominator, (float, int)):
        if denominator != 0:
            return numerator / denominator
        else:
            return zero_division
    else:
        mask = denominator == 0.0
        denominator[mask] = 1
        result = numerator / denominator
        if not mask.any():
            return result
        else:
            result[mask] = zero_division
            return result


def do_calculate_metric(
    confusion_ele_list: List[np.ndarray],
    metric_name: str,
    average: Union[Average, str] = "none",
    zero_division: int = 0,
):
    """
    Args:
        confusion_ele_list: the returned result of function ``cal_confusion_matrix_elements``.
        metric_name: the simplified metric name from function ``check_metric_name_and_unify``.
        average: type of averaging performed if not binary classification.
            Defaults to ``"macro"``.
            - ``"macro"``: calculate metrics for each label, and find their unweighted mean.
                This does not take label imbalance into account.
            - ``"weighted"``: calculate metrics for each label, and find their average
                weighted by support (the number of true instances for each label).
            - ``"micro"``: calculate metrics globally by considering each element of the label
                indicator matrix as a label.
            - ``"none"``: the scores for each class are returned.
        zero_division: the value to return when there is a zero division, for example, when all
            predictions and labels are negative. Defaults to 0.
    """
    ele_list: List[Union[np.ndarray, int, float]]
    metric = metric_name
    div_0 = zero_division
    # pre-process average
    average = Average(average)
    if len(confusion_ele_list[0].shape) == 0:
        average = Average.NONE  # for binary tasks, other average methods are meaningless.
        ele_list = [int(l) for l in confusion_ele_list]
    if average == Average.MICRO:
        ele_list = [int(l.sum()) for l in confusion_ele_list]
    else:
        ele_list = confusion_ele_list
    tp, tn, fp, fn, p, n = ele_list
    # calculate
    numerator: Union[np.ndarray, int, float]
    denominator: Union[np.ndarray, int, float]
    if metric == "tpr":
        numerator, denominator = tp, p
    elif metric == "tnr":
        numerator, denominator = tn, n
    elif metric == "ppv":
        numerator, denominator = tp, (tp + fp)
    elif metric == "npv":
        numerator, denominator = tn, (tn + fn)
    elif metric == "fnr":
        numerator, denominator = fn, p
    elif metric == "fpr":
        numerator, denominator = fp, n
    elif metric == "fdr":
        numerator, denominator = fp, (fp + tp)
    elif metric == "for":
        numerator, denominator = fn, (fn + tn)
    elif metric == "pt":
        tpr = handle_zero_divide(tp, p, div_0)
        tnr = handle_zero_divide(tn, n, div_0)
        numerator = np.sqrt(tpr * (1 - tnr)) + tnr - 1
        denominator = tpr + tnr - 1
    elif metric == "ts":
        numerator, denominator = tp, (tp + fn + fp)
    elif metric == "acc":
        numerator, denominator = (tp + tp), (p + n)
    elif metric == "ba":
        tpr = handle_zero_divide(tp, p, div_0)
        tnr = handle_zero_divide(tn, n, div_0)
        numerator, denominator = (tpr + tnr), 2
    elif metric == "f1":
        numerator, denominator = tp * 2, (tp * 2 + fn + fp)
    elif metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elif metric == "fm":
        tpr = handle_zero_divide(tp, p, div_0)
        ppv = handle_zero_divide(tp, (tp + fp), div_0)
        numerator = np.sqrt(ppv * tpr)
        denominator = 1
    elif metric == "bm":
        tpr = handle_zero_divide(tp, p, div_0)
        tnr = handle_zero_divide(tn, n, div_0)
        numerator = tpr + tnr - 1
        denominator = 1
    elif metric == "mk":
        ppv = handle_zero_divide(tp, (tp + fp), div_0)
        npv = handle_zero_divide(tn, (tn + fn), div_0)
        numerator = ppv + npv - 1
        denominator = 1
    else:
        raise NotImplementedError("the metric is not implemented.")
    result = handle_zero_divide(numerator, denominator, div_0)

    if average == Average.MICRO or average == Average.NONE:
        return result

    weights = None
    if average == Average.WEIGHTED:
        weights = p
    result = np.average(result, weights=weights)
    return result


def check_metric_name_and_unify(metric_name: str):
    """
    There are many metrics related to confusion matrix, and some of the metrics have
    more than one names. In addition, some of the names are very long.
    Therefore, this function is used to simplify the implementation.
    """
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["sensitivity", "recall", "hit_rate", "true_positive_rate", "tpr"]:
        return "tpr"
    elif metric_name in ["specificity", "selectivity", "true_negative_rate", "tnr"]:
        return "tnr"
    elif metric_name in ["precision", "positive_predictive_value", "ppv"]:
        return "ppv"
    elif metric_name in ["negative_predictive_value", "npv"]:
        return "npv"
    elif metric_name in ["miss_rate", "false_negative_rate", "fnr"]:
        return "fnr"
    elif metric_name in ["fall_out", "false_positive_rate", "fpr"]:
        return "fpr"
    elif metric_name in ["false_discovery_rate", "fdr"]:
        return "fdr"
    elif metric_name in ["false_omission_rate", "for"]:
        return "for"
    elif metric_name in ["prevalence_threshold", "pt"]:
        return "pt"
    elif metric_name in ["threat_score", "critical_success_index", "ts", "csi"]:
        return "ts"
    elif metric_name in ["accuracy", "acc"]:
        return "acc"
    elif metric_name in ["balanced_accuracy", "ba"]:
        return "ba"
    elif metric_name in ["f1_score", "f1"]:
        return "f1"
    elif metric_name in ["matthews_correlation_coefficient", "mcc"]:
        return "mcc"
    elif metric_name in ["fowlkes_mallows_index", "fm"]:
        return "fm"
    elif metric_name in ["informedness", "bookmaker_informedness", "bm"]:
        return "bm"
    elif metric_name in ["markedness", "deltap", "mk"]:
        return "mk"
    else:
        raise NotImplementedError("the metric is not implemented.")
