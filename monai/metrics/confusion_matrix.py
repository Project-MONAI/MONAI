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
from typing import Union

import torch

from monai.metrics.utils import *


class ConfusionMatrixMetric:
    """
    Compute confusion matrix related metrics. This function supports to calculate all metrics mentioned in:
    `Confusion matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`_.
    It can support both multi-classes and multi-labels classification and segmentation tasks.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` for an instance to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background so excluding it in such cases helps convergence.

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
        compute_sample: if ``True``, each sample's metric will be computed first. Defaults to ``False``.
        output_class: if ``True``, scores for each class will be returned. The final result is each class's score.
            If average these scores, you will get the macro average of each class. Otherwise, the micro average score
            of each class will be returned. Defaults to ``False``.

    """

    def __init__(
        self,
        include_background: bool = True,
        metric_name: str = "hit_rate",
        compute_sample: bool = False,
        output_class: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.metric_name = metric_name
        self.compute_sample = compute_sample
        self.output_class = output_class

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Args:
            y_pred: input data to compute. It must be one-hot format and first dim is batch.
                The values should be binarized.
            y: ground truth to compute the metric. It must be one-hot format and first dim is batch.
                The values should be binarized.
        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than two dimensions.
        """
        # check binarized input
        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn("y_pred is not a binarized tensor here!")
        if not torch.all(y.byte() == y):
            raise ValueError("y should be a binarized tensor.")
        # check dimension
        dims = y_pred.ndimension()
        if dims < 2:
            raise ValueError("y_pred should have at least two dimensions.")
        elif dims == 2 or (dims == 3 and y_pred.shape[-1] == 1):
            if self.compute_sample:
                warnings.warn("As for classification task, compute_sample should be False.")
                self.compute_sample = False

        confusion_matrix = get_confusion_matrix(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
        )

        if self.compute_sample:
            confusion_matrix = compute_confusion_matrix_metric(self.metric_name, confusion_matrix)
            if self.output_class:
                f, not_nans = do_metric_reduction(confusion_matrix, MetricReduction.MEAN_BATCH)
            else:
                f, not_nans = do_metric_reduction(confusion_matrix, MetricReduction.MEAN)
        else:
            if self.output_class:
                f, _ = do_metric_reduction(confusion_matrix, MetricReduction.SUM_BATCH)
            else:
                f, _ = do_metric_reduction(confusion_matrix, MetricReduction.SUM)
            f = compute_confusion_matrix_metric(self.metric_name, f)
            nans = torch.isnan(f)
            not_nans = (~nans).float()
            f[nans] = 0
            not_nans = not_nans.sum(dim=0)
        return f, not_nans


def get_confusion_matrix(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
):
    """
    Compute confusion matrix. A tensor with the shape [BC4] will be returned. Where, the third dimension
    represents the number of true positive, false positive, true negative and false negative values for
    each channel of each sample within the input batch. Where, B equals to the batch size and C equals to
    the number of classes that need to be computed.

    Args:
        y_pred: input data to compute. It must be one-hot format and first dim is batch.
            The values should be binarized.
        y: ground truth to compute the metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip metric computation on the first channel of
            the predicted output. Defaults to True.

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.
    """

    if not include_background:
        y_pred, y = ignore_background(
            y_pred=y_pred,
            y=y,
        )

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    # get confusion matrix related metric
    with torch.no_grad():
        batch_size, n_class = y_pred.shape[:2]
        # convert to [BNS], where S is the number of pixels for one sample.
        # As for classification tasks, S equals to 1.
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

    return torch.stack([tp, fp, tn, fn], dim=-1)


def compute_confusion_matrix_metric(metric_name: str, confusion_matrix: torch.Tensor):
    """
    This function is used to compute confusion matrix related metric.

    Args:
        metric_name: [``"sensitivity"``, ``"specificity"``, ``"precision"``, ``"negative predictive value"``,
            ``"miss rate"``, ``"fall out"``, ``"false discovery rate"``, ``"false omission rate"``,
            ``"prevalence threshold"``, ``"threat score"``, ``"accuracy"``, ``"balanced accuracy"``,
            ``"f1 score"``, ``"matthews correlation coefficient"``, ``"fowlkes mallows index"``,
            ``"informedness"``, ``"markedness"``]
            Some of the metrics have multiple aliases (as shown in the wikipedia page aforementioned),
            and you can also input those names instead.
        confusion_matrix: Please see the doc string of the function ``get_confusion_matrix`` for more details.

    Raises:
        ValueError: when the size of the last dimension of confusion_matrix is not 4.
        NotImplementedError: when specify a not implemented metric_name.

    """

    metric = check_confusion_matrix_metric_name(metric_name)

    input_dim = confusion_matrix.ndimension()
    if input_dim == 1:
        confusion_matrix = confusion_matrix.unsqueeze(dim=0)
    if confusion_matrix.shape[-1] != 4:
        raise ValueError("the size of the last dimension of confusion_matrix should be 4.")

    tp = confusion_matrix[..., 0]
    fp = confusion_matrix[..., 1]
    tn = confusion_matrix[..., 2]
    fn = confusion_matrix[..., 3]
    p = tp + fn
    n = fp + tn
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
        numerator, denominator = (tp + tn), (p + n)
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
        result = torch.where(denominator != 0, numerator / denominator, torch.tensor(float("nan")))
    else:
        result = numerator / denominator
    return result


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
