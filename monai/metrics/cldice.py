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

import torch

from monai.losses.cldice import soft_skel
from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction, look_up_option

from .metric import CumulativeIterationMetric

__all__ = ["ClDiceMetric", "compute_cldice"]


def _compute_converged_skel(img: torch.Tensor, iter_: int, reduce_axis: list[int]) -> torch.Tensor:
    """
    Compute the morphological soft skeleton with ``iter_`` iterations, and if any non-empty mask
    still yields zero skeleton mass (e.g. thick solid shapes), continue skeletonization up to
    the half-width of the spatial dimensions so the medial axis is extracted before falling back.

    Args:
        img: Binary or soft segmentation tensor of shape ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
        iter_: Initial number of morphological skeletonization iterations.
        reduce_axis: Spatial dimensions to sum over when checking mask and skeleton mass.

    Returns:
        Skeletonized tensor of the same shape as ``img``.
    """
    skel = soft_skel(img, iter_)
    img_sum = torch.sum(img, dim=reduce_axis)
    skel_sum = torch.sum(skel, dim=reduce_axis)
    needs_more = (img_sum > 0) & (skel_sum == 0)
    if not torch.any(needs_more):
        return skel

    max_iter = max(iter_, max(img.shape[2:]) // 2)
    if max_iter <= iter_:
        return skel

    converged_skel = soft_skel(img, max_iter)
    view_shape = img_sum.shape + (1,) * (img.ndim - 2)
    return torch.where(needs_more.reshape(view_shape), converged_skel, skel)


class ClDiceMetric(CumulativeIterationMetric):
    """
    Compute the centerline Dice (clDSC), Topology Precision (Tprec), or Topology Sensitivity (Tsens)
    score between predictions and reference segmentations for 2D and 3D inputs.

    Based on:
        Shit et al., "clDice - a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation", CVPR 2021.

    Args:
        include_background: Whether to include the first channel (background) in computation. Defaults to True.
        iter_: Number of iterations for morphological skeletonization. Defaults to 3.
        mode: Metric to return: ``"cldice"``, ``"tprec"``, or ``"tsens"``. Defaults to ``"cldice"``.
        reduction: Reduction mode across batches (``"mean"``, ``"sum"``, ``"none"``, etc.). Defaults to ``"mean"``.
        get_not_nans: Whether to return the ``not_nans`` count alongside the reduced metric. Defaults to False.
        smooth: Smoothing constant to avoid division by zero. Defaults to 1e-5.
    """

    def __init__(
        self,
        include_background: bool = True,
        iter_: int = 3,
        mode: str = "cldice",
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        """Initialize ClDiceMetric with skeletonization iterations, mode, and reduction settings."""
        super().__init__()
        self.include_background = include_background
        self.iter_ = iter_
        self.mode = look_up_option(mode.lower(), {"cldice", "tprec", "tsens"})
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.smooth = smooth

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the metric tensor for a single batch of predictions and targets.

        Args:
            y_pred: Predicted segmentation tensor of shape ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
            y: Reference segmentation tensor with the same shape as ``y_pred``.

        Returns:
            A tensor of shape ``[B, C]`` (or ``[B, C - 1]`` if ``include_background=False``) containing the scores.

        Raises:
            ValueError: When ``y_pred`` is not a 4D (2D) or 5D (3D) tensor, or shapes do not match.
        """
        if y_pred.ndim not in (4, 5):
            raise ValueError(f"y_pred should be a 4D (2D) or 5D (3D) tensor, got shape {y_pred.shape}.")
        return compute_cldice(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            iter_=self.iter_,
            mode=self.mode,
            smooth=self.smooth,
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate the buffered metric results across iterations according to the specified reduction mode.

        Args:
            reduction: Optional override for the reduction mode (``"mean"``, ``"sum"``, ``"none"``, etc.).

        Returns:
            The aggregated metric tensor, or a tuple ``(aggregated_metric, not_nans)`` if ``get_not_nans=True``.

        Raises:
            ValueError: When the buffered data to aggregate is not a PyTorch Tensor.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("The data to aggregate must be a PyTorch Tensor.")
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_cldice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    iter_: int = 3,
    mode: str = "cldice",
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Functional interface to compute ``cldice``, ``tprec``, or ``tsens`` per batch and per class.

    If a reference mask channel is empty while the corresponding prediction is non-empty, ``NaN`` is returned
    for that channel; if both reference and prediction masks are empty, ``1.0`` is returned. For thick masks
    where ``iter_`` is insufficient to reach the medial axis, skeletonization converges automatically before
    falling back to mask overlap if skeleton mass remains zero.

    Args:
        y_pred: Predicted segmentation tensor of shape ``[B, C, H, W]`` or ``[B, C, H, W, D]``.
        y: Reference segmentation tensor with the same shape as ``y_pred``.
        include_background: Whether to include the first channel (background) in computation. Defaults to True.
        iter_: Number of iterations for morphological skeletonization. Defaults to 3.
        mode: Metric to return: ``"cldice"``, ``"tprec"``, or ``"tsens"``. Defaults to ``"cldice"``.
        smooth: Smoothing constant for numerical stability. Defaults to 1e-5.

    Returns:
        A tensor of shape ``[batch_size, num_classes]`` containing the computed metric values.

    Raises:
        ValueError: When ``mode`` is unsupported or ``y_pred`` and ``y`` have different shapes.
    """
    mode = look_up_option(mode.lower(), {"cldice", "tprec", "tsens"})

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    reduce_axis = list(range(2, y_pred.ndim))
    skel_pred = _compute_converged_skel(y_pred, iter_, reduce_axis)
    skel_true = _compute_converged_skel(y, iter_, reduce_axis)

    y_sum = torch.sum(y, dim=reduce_axis)
    y_pred_sum = torch.sum(y_pred, dim=reduce_axis)
    skel_pred_sum = torch.sum(skel_pred, dim=reduce_axis)
    skel_true_sum = torch.sum(skel_true, dim=reduce_axis)
    intersection = torch.sum(y_pred * y, dim=reduce_axis)

    fallback_tprec = torch.where(y_pred_sum > 0, intersection / y_pred_sum, torch.zeros_like(y_pred_sum))
    fallback_tsens = torch.where(y_sum > 0, intersection / y_sum, torch.zeros_like(y_sum))

    skel_pred_overlap = torch.sum(skel_pred * y, dim=reduce_axis)
    skel_true_overlap = torch.sum(skel_true * y_pred, dim=reduce_axis)

    tprec = torch.where(
        skel_pred_sum > 0,
        torch.where(
            skel_pred_overlap > 0,
            (skel_pred_overlap + smooth) / (skel_pred_sum + smooth),
            torch.zeros_like(skel_pred_sum),
        ),
        fallback_tprec,
    )
    tsens = torch.where(
        skel_true_sum > 0,
        torch.where(
            skel_true_overlap > 0,
            (skel_true_overlap + smooth) / (skel_true_sum + smooth),
            torch.zeros_like(skel_true_sum),
        ),
        fallback_tsens,
    )

    if mode == "tprec":
        score = tprec
    elif mode == "tsens":
        score = tsens
    else:
        denom = tprec + tsens
        score = torch.where(denom > 0, (2.0 * tprec * tsens) / denom, torch.zeros_like(denom))

    nan_tensor = torch.tensor(float("nan"), device=y.device, dtype=y.dtype)
    ones_tensor = torch.tensor(1.0, device=y.device, dtype=y.dtype)
    return torch.where(y_sum > 0, score, torch.where(y_pred_sum > 0, nan_tensor, ones_tensor))
