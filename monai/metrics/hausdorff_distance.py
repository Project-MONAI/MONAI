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

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from monai.metrics.utils import (
    compute_voronoi_regions_fast,
    do_metric_reduction,
    get_edge_surface_distance,
    ignore_background,
    prepare_spacing,
)
from monai.utils import MetricReduction, convert_data_type

from .metric import CumulativeIterationMetric

__all__ = ["HausdorffDistanceMetric", "compute_hausdorff_distance"]


class HausdorffDistanceMetric(CumulativeIterationMetric):
    """
    Compute Hausdorff Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both directed and non-directed Hausdorff distance calculation. In addition, specify the `percentile`
    parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    The ``per_component=True`` approach computes the Hausdorff distance on a per-connected component basis in the ground
    truth segmentation. This ensures that each component contributes equally to the final metric, regardless of its size.
    Traditional Hausdorff distance can be dominated by large structures, but the per-component method gives a more
    balanced evaluation, particularly for small or fragmented objects. This provides a granular assessment of segmentation
    quality, which is especially important in cases with multiple disconnected foreground components.
    Note:
    - The input prediction (`y_pred`) and ground truth (`y`) must both have 2 channels (foreground/background),
    with binary segmentation (0 for background, 1 for foreground). That is, this assumes the shape of both prediction
    and ground truth is B2HW[D].
    - This method cannot be used with multiclass segmentation.
    For more information, refer to the original paper: https://arxiv.org/abs/2410.18684

    Args:
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
        per_component: whether to compute the Hausdorff distance on a per-connected component basis. Defaults to ``False``.

    """

    def __init__(
        self,
        include_background: bool = False,
        distance_metric: str = "euclidean",
        percentile: float | None = None,
        directed: bool = False,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        per_component: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.directed = directed
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.per_component = per_component

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor, **kwargs: Any) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.
            kwargs: additional parameters, e.g. ``spacing`` should be passed to correctly compute the metric.
                ``spacing``: spacing of pixel (or voxel). This parameter is relevant only
                if ``distance_metric`` is set to ``"euclidean"``.
                If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
                the length of the sequence must be equal to the image dimensions.
                This spacing will be used for all images in the batch.
                If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
                If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
                else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
                for all images in batch. Defaults to ``None``.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        if self.per_component:
            if y_pred.ndim not in (4, 5) or y.ndim not in (4, 5) or y_pred.shape[1] != 2 or y.shape[1] != 2:
                same_rank = y_pred.ndim == y.ndim and y_pred.ndim in (4, 5)
                binary_channels = y_pred.shape[1] == 2 and y.shape[1] == 2
                same_shape = y_pred.shape == y.shape
                if not (same_rank and binary_channels and same_shape):
                    raise ValueError(
                        "per_component requires matching 4D/5D binary tensors "
                        "(B, 2, H, W) or (B, 2, D, H, W). "
                        f"Got y_pred={tuple(y_pred.shape)}, y={tuple(y.shape)}."
                    )
        # compute (BxC) for each channel for each batch
        return compute_hausdorff_distance(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
            spacing=kwargs.get("spacing"),
            per_component=self.per_component,
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_hausdorff_distance(
    y_pred: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: float | None = None,
    directed: bool = False,
    spacing: int | float | np.ndarray | Sequence[int | float | np.ndarray | Sequence[int | float]] | None = None,
    per_component: bool = False,
) -> torch.Tensor:
    """
    Compute the Hausdorff distance.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        spacing: spacing of pixel (or voxel). This parameter is relevant only if ``distance_metric`` is set to ``"euclidean"``.
            If a single number, isotropic spacing with that value is used for all images in the batch. If a sequence of numbers,
            the length of the sequence must be equal to the image dimensions. This spacing will be used for all images in the batch.
            If a sequence of sequences, the length of the outer sequence must be equal to the batch size.
            If inner sequence has length 1, isotropic spacing with that value is used for all images in the batch,
            else the inner sequence length must be equal to the image dimensions. If ``None``, spacing of unity is used
            for all images in batch. Defaults to ``None``.
        per_component: whether to compute the Hausdorff distance on a per-connected component basis. Defaults to ``False``.
    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)
    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    batch_size, n_class = y_pred.shape[:2]
    hd = torch.empty((batch_size, n_class), dtype=torch.float, device=y_pred.device)

    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)

    for b, c in np.ndindex(batch_size, n_class):
        if per_component:
            if y[b, c].sum() == 0 or y_pred[b, c].sum() == 0:
                if y_pred[b, c].sum() == 0 and y[b, c].sum() == 0:
                    hd[b, c] = 0.0
                else:
                    hd[b, c] = 1.0
                continue
            cc_assignment = compute_voronoi_regions_fast(y_pred[b, c].cpu().numpy())
            if cc_assignment.device != y_pred[b, c].device:
                cc_assignment = cc_assignment.to(y_pred[b, c].device)
            max_list = []
            for cc_id in torch.unique(cc_assignment.view(-1)):
                cc_mask = cc_assignment == cc_id

                coords = torch.nonzero(cc_mask, as_tuple=False)
                min_corner_idx = coords.min(dim=0).values
                max_corner_idx = coords.max(dim=0).values

                crop_pred = (
                    y_pred[b, c][
                        min_corner_idx[0] : max_corner_idx[0] + 1,
                        min_corner_idx[1] : max_corner_idx[1] + 1,
                        min_corner_idx[2] : max_corner_idx[2] + 1,
                    ]
                    if y_pred.ndim == 5
                    else y_pred[b, c][
                        min_corner_idx[0] : max_corner_idx[0] + 1, min_corner_idx[1] : max_corner_idx[1] + 1
                    ]
                )

                crop_label = (
                    y[b, c][
                        min_corner_idx[0] : max_corner_idx[0] + 1,
                        min_corner_idx[1] : max_corner_idx[1] + 1,
                        min_corner_idx[2] : max_corner_idx[2] + 1,
                    ]
                    if y.ndim == 5
                    else y[b, c][min_corner_idx[0] : max_corner_idx[0] + 1, min_corner_idx[1] : max_corner_idx[1] + 1]
                )

                cc_crop_mask = (
                    cc_mask[
                        min_corner_idx[0] : max_corner_idx[0] + 1,
                        min_corner_idx[1] : max_corner_idx[1] + 1,
                        min_corner_idx[2] : max_corner_idx[2] + 1,
                    ]
                    if y_pred.ndim == 5
                    else cc_mask[min_corner_idx[0] : max_corner_idx[0] + 1, min_corner_idx[1] : max_corner_idx[1] + 1]
                )

                pred_masked = crop_pred * cc_crop_mask
                label_masked = crop_label * cc_crop_mask

                _, distances, _ = get_edge_surface_distance(
                    pred_masked,
                    label_masked,
                    distance_metric=distance_metric,
                    spacing=spacing_list[b],
                    symmetric=not directed,
                    class_index=c,
                )
                percentile_distances = [_compute_percentile_hausdorff_distance(d, percentile) for d in distances]
                max_list.append(torch.stack(percentile_distances))

            hd[b, c] = torch.nanmean(torch.stack(max_list)) if max_list else 0.0
        else:
            _, distances, _ = get_edge_surface_distance(
                y_pred[b, c],
                y[b, c],
                distance_metric=distance_metric,
                spacing=spacing_list[b],
                symmetric=not directed,
                class_index=c,
            )
            percentile_distances = [_compute_percentile_hausdorff_distance(d, percentile) for d in distances]
            max_distance = torch.max(torch.stack(percentile_distances))
            hd[b, c] = max_distance
    return hd


def _compute_percentile_hausdorff_distance(
    surface_distance: torch.Tensor, percentile: float | None = None
) -> torch.Tensor:
    """
    This function is used to compute the Hausdorff distance.
    """

    # for both pred and gt do not have foreground
    if surface_distance.shape == (0,):
        return torch.tensor(np.nan, dtype=torch.float, device=surface_distance.device)

    if not percentile:
        return surface_distance.max()

    if 0 <= percentile <= 100:
        return torch.quantile(surface_distance, percentile / 100)
    raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")
