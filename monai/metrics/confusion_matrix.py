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

from monai.metrics.seg_metric_utils import *


class ConfusionMatrixMetric:
    """
    Compute confusion matrix related metrics. This function supports to calculate all metrics mentioned in:
    `Confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_.
    It can support both multi-classes and multi-labels segmentation tasks.

    Args:
        include_background: whether to skip metric computation on the first channel of
            the predicted output. Defaults to True.
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
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.

    """

    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        activation: Optional[Union[str, Callable]] = None,
        bin_mode: Optional[str] = "threshold",
        bin_threshold: Union[float, Sequence[float]] = 0.5,
        metric_name: str = "hit_rate",
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.activation = activation
        self.bin_mode = bin_mode
        self.bin_threshold = bin_threshold
        self.metric_name = metric_name
        self.reduction = reduction

        self.not_nans: Optional[torch.Tensor] = None  # keep track for valid elements in the batch

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Args:
            y_pred: input data to compute. It must be one-hot format and first dim is batch.
            y: ground truth, the first dim is batch.

        """

        # compute metric (BxC) for each channel for each batch.
        f = compute_confusion_matrix_metric(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            to_onehot_y=self.to_onehot_y,
            activation=self.activation,
            bin_mode=self.bin_mode,
            bin_threshold=self.bin_threshold,
            metric_name=self.metric_name,
        )

        # some elements might be Nan, we need to account for it.
        nans = torch.isnan(f)
        not_nans = (~nans).float()

        # save not_nans since we may need it later to know how many elements were valid
        self.not_nans = not_nans

        # do metric reduction
        f = do_metric_reduction(f, self.reduction)
        return f


def compute_confusion_matrix_metric(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    to_onehot_y: bool = False,
    activation: Optional[Union[str, Callable]] = None,
    bin_mode: Optional[str] = "threshold",
    bin_threshold: Union[float, Sequence[float]] = 0.5,
    metric_name: str = "hit_rate",
):
    """
    Compute confusion matrix related metrics. This function supports to calculate all metrics
    mentioned in: `Confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            it must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
        y: ground truth, the first dim is batch.
            example shape: [16, 1, 32, 32] will be converted into [16, 3, 32, 32].
            alternative shape: [16, 3, 32, 32] and set `to_onehot_y=False` to use 3-class labels directly.
        include_background: whether to skip metric computation on the first channel of
            the predicted output. Defaults to True.
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

    Raises:
        AssertionError: when `y_pred` and `y` have different shapes.
        AssertionError: when `y_pred` has less than three dimensions.
        NotImplementedError: when the metric is not implemented.
    """

    y_pred, y = preprocess_input(
        y_pred=y_pred,
        y=y,
        to_onehot_y=to_onehot_y,
        activation=activation,
        bin_mode=bin_mode,
        bin_threshold=bin_threshold,
        include_background=include_background,
    )

    # get confusion matrix related metric
    with torch.no_grad():
        dims = y_pred.ndimension()
        assert dims > 2, "for segmentation task, y_pred should have at least three dimensions."
        batch_size, n_class = y_pred.shape[:2]
        # convert to [BNS], where S is the number of pixels for one sample.
        y_pred = y_pred.view(batch_size, n_class, -1)
        y = y.view(batch_size, n_class, -1)
        tp = ((y_pred + y) == 2).float()
        tn = ((y_pred + y) == 0).float()

        tp = tp.sum(dim=[2])
        tn = tn.sum(dim=[2])
        p = y.sum(dim=[2])
        n = y.shape[-1] - p

        fn = p - tp
        fp = n - tn

    metric = check_confusion_matrix_metric_name(metric_name)

    # calculate metric
    numerator: torch.Tensor
    denominator: Union[torch.Tensor, float]
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
        tpr = torch.where(p > 0, tp / p, torch.tensor(float("nan")))
        tnr = torch.where(n > 0, tn / n, torch.tensor(float("nan")))
        numerator = torch.sqrt(tpr * (1.0 - tnr)) + tnr - 1.0
        denominator = tpr + tnr - 1.0
    elif metric == "ts":
        numerator, denominator = tp, (tp + fn + fp)
    elif metric == "acc":
        numerator, denominator = (tp + tp), (p + n)
    elif metric == "ba":
        tpr = torch.where(p > 0, tp / p, torch.tensor(float("nan")))
        tnr = torch.where(n > 0, tn / n, torch.tensor(float("nan")))
        numerator, denominator = (tpr + tnr), 2.0
    elif metric == "f1":
        numerator, denominator = tp * 2.0, (tp * 2.0 + fn + fp)
    elif metric == "mcc":
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    elif metric == "fm":
        tpr = torch.where(p > 0, tp / p, torch.tensor(float("nan")))
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), torch.tensor(float("nan")))
        numerator = torch.sqrt(ppv * tpr)
        denominator = 1.0
    elif metric == "bm":
        tpr = torch.where(p > 0, tp / p, torch.tensor(float("nan")))
        tnr = torch.where(n > 0, tn / n, torch.tensor(float("nan")))
        numerator = tpr + tnr - 1.0
        denominator = 1.0
    elif metric == "mk":
        ppv = torch.where((tp + fp) > 0, tp / (tp + fp), torch.tensor(float("nan")))
        npv = torch.where((tn + fn) > 0, tn / (tn + fn), torch.tensor(float("nan")))
        npv = tn / (tn + fn)
        numerator = ppv + npv - 1.0
        denominator = 1.0
    else:
        raise NotImplementedError("the metric is not implemented.")

    if isinstance(denominator, torch.Tensor):
        result = torch.where(denominator > 0, numerator / denominator, torch.tensor(float("nan")))
    else:
        result = numerator / denominator

    return result  # returns array of metric with shape: [batch, n_classes]


def check_confusion_matrix_metric_name(metric_name: str):
    """
    There are many metrics related to confusion matrix, and some of the metrics have
    more than one names. In addition, some of the names are very long.
    Therefore, this function is used to check and simplify the name.

    Returns:
        Simplified metric name.

    Raises:
        NotImplementedError: when the metric is not implemented.
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
