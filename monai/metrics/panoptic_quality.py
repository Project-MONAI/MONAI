# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Union

import torch

from monai.metrics.metric import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction, remap_instance_id
from monai.utils import MetricReduction, optional_import

linear_sum_assignment, _ = optional_import("scipy.optimize", name="linear_sum_assignment")

__all__ = ["PanopticQualityMetric", "compute_panoptic_quality"]


class PanopticQualityMetric(CumulativeIterationMetric):
    """
    Compute Panoptic Quality between two tensors. If specifying `metric_name` to "SQ" or "RQ",
    Segmentation Quality (SQ) or Recognition Quality (RQ) will be returned instead.

    Panoptic Quality is a metric used in panoptic segmentation tasks. This task unifies the typically distinct tasks
    of semantic segmentation (assign a class label to each pixel) and
    instance segmentation (detect and segment each object instance). Compared with semantic segmentation, panoptic
    segmentation distinguish different instances that belong to same class.
    Compared with instance segmentation, panoptic segmentation does not allow overlap and only one semantic label and
    one instance id can be assigned to each pixel.
    Please refer to the following paper for more details:
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf

    Different from the original panoptic quality, this class refers to the evaluation metric of CoNiC Challenge 2022,
    a Nuclei instance segmentation and classification task. The referred implementation can be found in:
    https://github.com/TissueImageAnalytics/CoNIC

    Args:
        num_classes: number of classes. The number should not count the background.
        metric_name: output metric. The value can be "pq", "sq" or "rq".
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.
        match_iou: IOU threshould to determine the pairing between `y_pred` and `y`. Usually,
            it should >= 0.5, the pairing between instances of `y_pred` and `y` are identical.
            If set `match_iou` < 0.5, this function uses Munkres assignment to find the
            maximal amout of unique pairing.
        smooth_nr: a small constant added to the numerator to avoid zero.

    """

    def __init__(
        self,
        num_classes: int,
        metric_name: str = "pq",
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN_BATCH,
        match_iou: float = 0.5,
        smooth_nr: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.match_iou = match_iou
        self.smooth_nr = smooth_nr
        self.metric_name = _check_panoptic_metric_name(metric_name)

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: Predictions. It must be in the form of B2HW and have integer type. The first channel and the
                second channel represent the instance predictions and classification predictions respectively.
            y: ground truth. It must have the same shape as `y_pred` and have integer type. The first channel and the
                second channel represent the instance labels and classification labels respectively.
                Values in the second channel of `y_pred` and `y` should be in the range of 0 to `self.num_classes`,
                where 0 represents the background.

        Raises:
            ValueError: when `y_pred` and `y` have different shapes.
            ValueError: when `y_pred` and `y` have != 2 channels.
            ValueError: when `y_pred` and `y` have != 4 dimensions.

        """
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

        if y_pred.shape[1] != 2:
            raise ValueError(
                f"for panoptic quality calculation, only 2 channels input is supported, got {y_pred.shape[1]}."
            )

        dims = y_pred.ndimension()
        if dims != 4:
            raise ValueError(f"y_pred should have 4 dimensions (batch, 2, h, w), got {dims}.")

        batch_size = y_pred.shape[0]

        outputs = torch.zeros([batch_size, self.num_classes, 4], device=y_pred.device)

        for b in range(batch_size):
            true_instance, pred_instance = y[b, 0], y_pred[b, 0]
            true_class, pred_class = y[b, 1], y_pred[b, 1]
            for c in range(self.num_classes):
                pred_instance_c = (pred_class == c + 1) * pred_instance
                true_instance_c = (true_class == c + 1) * true_instance

                outputs[b, c] = compute_panoptic_quality(
                    pred=pred_instance_c,
                    gt=true_instance_c,
                    remap=True,
                    match_iou=self.match_iou,
                    output_confusion_matrix=True,
                )

        return outputs

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
        """
        Execute reduction logic for the output of `compute_panoptic_quality`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, _ = do_metric_reduction(data, reduction or self.reduction)

        tp, fp, fn, iou_sum = f[..., 0], f[..., 1], f[..., 2], f[..., 3]
        if self.metric_name == "rq":
            return tp / (tp + 0.5 * fp + 0.5 * fn + self.smooth_nr)
        if self.metric_name == "sq":
            return iou_sum / (tp + self.smooth_nr)
        return iou_sum / (tp + 0.5 * fp + 0.5 * fn + self.smooth_nr)


def compute_panoptic_quality(
    pred: torch.Tensor,
    gt: torch.Tensor,
    metric_name: str = "pq",
    remap: bool = True,
    match_iou: float = 0.5,
    smooth_nr: float = 1e-6,
    output_confusion_matrix: bool = False,
):
    """Computes Panoptic Quality (PQ). If specifying `metric_name` to "SQ" or "RQ",
    Segmentation Quality (SQ) or Recognition Quality (RQ) will be returned instead.

    In addition, if `output_confusion_matrix` is True, the function will return a tensor with shape 4, which
    represents the true positive, false positive, false negative and the sum of iou. These four values are used to
    calculate PQ, and return them directly is able for further calculation over all images.

    Args:
        pred: input data to compute, it must be in the form of HW and have integer type.
        gt: ground truth. It must have the same shape as `pred` and have integer type.
        metric_name: output metric. The value can be "pq", "sq" or "rq".
        remap: whether to remap `pred` and `gt` to ensure contiguous ordering of instance id.
        match_iou: IOU threshould to determine the pairing between `pred` and `gt`. Usually,
            it should >= 0.5, the pairing between instances of `pred` and `gt` are identical.
            If set `match_iou` < 0.5, this function uses Munkres assignment to find the
            maximal amout of unique pairing.
        smooth_nr: a small constant added to the numerator to avoid zero.

    Raises:
        ValueError: when `pred` and `gt` have different shapes.
        ValueError: when `match_iou` <= 0.0 or > 1.0.

    """

    if gt.shape != pred.shape:
        raise ValueError(f"pred and gt should have same shapes, got {pred.shape} and {gt.shape}.")
    if match_iou <= 0.0 or match_iou > 1.0:
        raise ValueError(f"'match_iou' should be within (0, 1], got: {match_iou}.")

    gt = gt.int()
    pred = pred.int()

    if remap is True:
        gt = remap_instance_id(gt)
        pred = remap_instance_id(pred)

    pairwise_iou, true_id_list, pred_id_list = _get_pairwise_iou(pred, gt, device=pred.device)
    paired_iou, paired_true, paired_pred = _get_paired_iou(pairwise_iou, match_iou, device=pairwise_iou.device)

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp, fp, fn = len(paired_true), len(unpaired_pred), len(unpaired_true)
    iou_sum = paired_iou.sum()

    if output_confusion_matrix:
        return torch.as_tensor([tp, fp, fn, iou_sum], device=pred.device)

    metric_name = _check_panoptic_metric_name(metric_name)
    if metric_name == "rq":
        return torch.as_tensor(tp / (tp + 0.5 * fp + 0.5 * fn + smooth_nr), device=pred.device)
    if metric_name == "sq":
        return torch.as_tensor(iou_sum / (tp + smooth_nr), device=pred.device)
    return torch.as_tensor(iou_sum / (tp + 0.5 * fp + 0.5 * fn + smooth_nr), device=pred.device)


def _get_id_list(gt: torch.Tensor):
    id_list = list(gt.unique())
    # ensure id 0 is included
    if 0 not in id_list:
        id_list.insert(0, torch.tensor(0).int())

    return id_list


def _get_pairwise_iou(pred: torch.Tensor, gt: torch.Tensor, device: Union[str, torch.device] = "cpu"):
    pred_id_list = _get_id_list(pred)
    true_id_list = _get_id_list(gt)

    pairwise_iou = torch.zeros([len(true_id_list) - 1, len(pred_id_list) - 1], dtype=torch.float, device=device)
    true_masks: List[torch.Tensor] = []
    pred_masks: List[torch.Tensor] = []

    for t in true_id_list[1:]:
        t_mask = torch.as_tensor(gt == t, device=device).int()
        true_masks.append(t_mask)

    for p in pred_id_list[1:]:
        p_mask = torch.as_tensor(pred == p, device=device).int()
        pred_masks.append(p_mask)

    for true_id in range(1, len(true_id_list)):
        t_mask = true_masks[true_id - 1]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = list(pred_true_overlap.unique())
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:
                continue
            p_mask = pred_masks[pred_id - 1]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    return pairwise_iou, true_id_list, pred_id_list


def _get_paired_iou(pairwise_iou: torch.Tensor, match_iou: float = 0.5, device: Union[str, torch.device] = "cpu"):
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = torch.nonzero(pairwise_iou)[:, 0], torch.nonzero(pairwise_iou)[:, 1]
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1
        paired_pred += 1

        return paired_iou, paired_true, paired_pred

    pairwise_iou = pairwise_iou.cpu().numpy()
    paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
    paired_iou = pairwise_iou[paired_true, paired_pred]
    paired_true = torch.as_tensor(list(paired_true[paired_iou > match_iou] + 1), device=device)
    paired_pred = torch.as_tensor(list(paired_pred[paired_iou > match_iou] + 1), device=device)
    paired_iou = paired_iou[paired_iou > match_iou]

    return paired_iou, paired_true, paired_pred


def _check_panoptic_metric_name(metric_name: str):
    metric_name = metric_name.replace(" ", "_")
    metric_name = metric_name.lower()
    if metric_name in ["panoptic_quality", "pq"]:
        return "pq"
    if metric_name in ["segmentation_quality", "sq"]:
        return "sq"
    if metric_name in ["recognition_quality", "rq"]:
        return "rq"
    raise ValueError(f"metric name: {metric_name} is wrong, please use 'pq', 'sq' or 'rq'.")
