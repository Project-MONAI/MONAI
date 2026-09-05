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

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from monai.metrics.utils import do_metric_reduction, ignore_background, is_binary_tensor
from monai.utils import MetricReduction, optional_import

from .metric import CumulativeIterationMetric

__all__ = ["CenterlineDiceMetric", "compute_centerline_dice"]

skeletonize, has_skimage = optional_import("skimage.morphology", name="skeletonize")

_METRIC_NAMES = {
    "cldsc": "cldsc",
    "centerline_dice": "cldsc",
    "topology_precision": "topology_precision",
    "tprec": "topology_precision",
    "topology_sensitivity": "topology_sensitivity",
    "tsens": "topology_sensitivity",
}


class CenterlineDiceMetric(CumulativeIterationMetric):
    """
    Compute the centreline Dice score (clDSC), topology precision, or topology sensitivity between
    predicted and reference segmentation masks, based on skeletonization of the binary segmentations.
    These metrics evaluate the connectivity of segmentations and are particularly useful for tubular
    structures such as vessels.

    This implements the metrics defined in:

        Shit et al. "clDice -- a Novel Topology-Preserving Loss Function for Tubular Structure
        Segmentation." CVPR 2021. https://arxiv.org/abs/2003.07311

    With :math:`S_{Pred}` and :math:`S_{Ref}` the skeletons of the prediction and the reference
    segmentation, the topology precision and topology sensitivity are defined as:

    .. math::
        T_{prec}(S_{Pred}, Ref) = \\frac{|S_{Pred} \\cap Ref|}{|S_{Pred}|}, \\qquad
        T_{sens}(S_{Ref}, Pred) = \\frac{|S_{Ref} \\cap Pred|}{|S_{Ref}|}

    and the centreline Dice score is their harmonic mean:

    .. math::
        clDSC = \\frac{2 \\cdot T_{prec} \\cdot T_{sens}}{T_{prec} + T_{sens}}

    The class- and batch sample-wise values can be aggregated with the function `aggregate`.

    The skeletons are computed with ``skimage.morphology.skeletonize`` (2D inputs) and its
    ``method="lee"`` variant (3D inputs), matching the reference implementation of the paper
    (https://github.com/jocpae/clDice). ``scikit-image`` is an optional dependency imported with
    ``optional_import`` and is required to use this metric.

    The inputs ``y_pred`` and ``y`` are expected to be binarized one-hot tensors with shape
    BCHW[D]. If they contain continuous values (e.g. sigmoid outputs), binarize them first with a
    suitable threshold transform.

    Example:
        .. code-block:: python

            import torch
            from monai.metrics import CenterlineDiceMetric

            y_pred = torch.randint(0, 2, (4, 3, 64, 64)).float()
            y = torch.randint(0, 2, (4, 3, 64, 64)).float()

            metric = CenterlineDiceMetric(metric_name="cldsc", include_background=False)
            metric(y_pred, y)           # accumulate
            result = metric.aggregate() # shape: (n_classes - 1,) after mean reduction
            metric.reset()

    Args:
        metric_name: which metric to compute. One of ``"cldsc"`` (alias ``"centerline_dice"``),
            ``"topology_precision"`` (alias ``"tprec"``), ``"topology_sensitivity"`` (alias
            ``"tsens"``). Defaults to ``"cldsc"``.
        include_background: whether to include the first channel (index 0), which is by convention
            assumed to be background, in the computation. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan`
            values, available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``,
            ``"sum_batch"``, ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``.
            If "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count. Defaults to ``False``.
            `not_nans` is the number of batch samples for which not all class-specific values were
            nan values. If set to ``True``, the function `aggregate` will return both the aggregated
            value and the `not_nans` count.

    Raises:
        ValueError: if ``metric_name`` is not one of the supported metrics.
    """

    def __init__(
        self,
        metric_name: str = "cldsc",
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        if metric_name not in _METRIC_NAMES:
            raise ValueError(f"metric_name must be one of {sorted(_METRIC_NAMES)}, got '{metric_name}'.")
        self.metric_name = _METRIC_NAMES[metric_name]
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: binarized prediction tensor, shape BCHW[D].
            y: binarized reference tensor, shape BCHW[D].

        Raises:
            ValueError: when ``y_pred`` or ``y`` are not one-hot encoded BCHW[D] tensors.
        """
        return compute_centerline_dice(
            y_pred=y_pred, y=y, metric_name=self.metric_name, include_background=self.include_background
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregates the output of `_compute_tensor`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on
                `not-nan` values, available reduction modes: {``"none"``, ``"mean"``, ``"sum"``,
                ``"mean_batch"``, ``"sum_batch"``, ``"mean_channel"``, ``"sum_channel"``},
                default to `self.reduction`. if "none", will not do reduction.

        Returns:
            If `get_not_nans` is set to ``True``, this function returns the aggregated value and the
            `not_nans` count, otherwise only the aggregated value.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be a PyTorch Tensor.")

        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def _skeletonize(mask: np.ndarray) -> np.ndarray:
    """Compute the binary skeleton of a 2D or 3D mask (Zhang-Suen for 2D, Lee et al. for 3D)."""
    if not mask.any():
        return mask
    if mask.ndim == 2:
        return skeletonize(mask)
    return skeletonize(mask, method="lee")


def compute_centerline_dice(
    y_pred: torch.Tensor, y: torch.Tensor, metric_name: str = "cldsc", include_background: bool = True
) -> torch.Tensor:
    """
    Compute the centreline Dice score (clDSC), topology precision, or topology sensitivity for a
    batch of binarized segmentations. See :py:class:`CenterlineDiceMetric` for the definitions.

    In the case of a class being present neither in the predicted segmentation nor in the reference
    segmentation, a nan value is returned for this class. In the case of a class being present in
    only one of predicted segmentation or reference segmentation, 0 is returned for this class
    (a missing prediction cannot cover the reference skeleton and vice versa).

    Args:
        y_pred: binarized prediction tensor, one-hot encoded, shape BCHW[D].
        y: binarized reference tensor, one-hot encoded, shape BCHW[D].
        metric_name: which metric to compute. One of ``"cldsc"`` (alias ``"centerline_dice"``),
            ``"topology_precision"`` (alias ``"tprec"``), ``"topology_sensitivity"`` (alias
            ``"tsens"``). Defaults to ``"cldsc"``.
        include_background: whether to include the first channel (background) in the computation.
            Defaults to ``True``.

    Returns:
        PyTorch tensor of shape [B, C] with the metric value for each batch sample and class.

    Raises:
        ValueError: if ``y_pred`` and/or ``y`` are not PyTorch tensors, do not have four or five
            dimensions, have different shapes, or if ``metric_name`` is unknown.
    """
    if metric_name not in _METRIC_NAMES:
        raise ValueError(f"metric_name must be one of {sorted(_METRIC_NAMES)}, got '{metric_name}'.")
    metric = _METRIC_NAMES[metric_name]

    if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("y_pred and y must be PyTorch Tensor.")

    if y_pred.ndimension() not in (4, 5) or y.ndimension() not in (4, 5):
        raise ValueError("y_pred and y should be one-hot encoded: [B,C,H,W] or [B,C,H,W,D].")

    if y_pred.shape != y.shape:
        raise ValueError(
            f"y_pred and y should have same shape, but instead, shapes are {y_pred.shape} (y_pred) and {y.shape} (y)."
        )

    is_binary_tensor(y_pred, "y_pred")
    is_binary_tensor(y, "y")

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    batch_size, n_class = y_pred.shape[:2]
    result = torch.empty((batch_size, n_class), device=y_pred.device, dtype=torch.float)

    y_pred_np = y_pred.detach().cpu().numpy() > 0
    y_np = y.detach().cpu().numpy() > 0

    for b, c in np.ndindex(batch_size, n_class):
        pred, ref = y_pred_np[b, c], y_np[b, c]
        if not ref.any() and not pred.any():
            # the class is neither present in the prediction nor in the reference segmentation
            result[b, c] = torch.tensor(np.nan)
            continue
        skel_pred = _skeletonize(pred)
        skel_ref = _skeletonize(ref)
        n_skel_pred, n_skel_ref = int(skel_pred.sum()), int(skel_ref.sum())
        # an empty prediction cannot cover any reference skeleton (and vice versa): score 0
        t_prec = float(np.logical_and(skel_pred, ref).sum()) / n_skel_pred if n_skel_pred > 0 else 0.0
        t_sens = float(np.logical_and(skel_ref, pred).sum()) / n_skel_ref if n_skel_ref > 0 else 0.0
        if metric == "topology_precision":
            result[b, c] = t_prec
        elif metric == "topology_sensitivity":
            result[b, c] = t_sens
        else:
            result[b, c] = 0.0 if (t_prec + t_sens) == 0 else 2.0 * t_prec * t_sens / (t_prec + t_sens)

    return result
