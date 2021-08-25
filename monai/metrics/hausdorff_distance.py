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

import warnings
from typing import Optional, Union

import numpy as np
import torch

from monai.metrics.utils import do_metric_reduction, get_mask_edges, get_surface_distance, ignore_background
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric

__all__ = ["HausdorffDistanceMetric", "compute_hausdorff_distance", "compute_percent_hausdorff_distance"]


class HausdorffDistanceMetric(CumulativeIterationMetric):
    """
    Compute Hausdorff Distance between two tensors. It can support both multi-classes and multi-labels tasks.
    It supports both directed and non-directed Hausdorff distance calculation. In addition, specify the `percentile`
    parameter can get the percentile of the distance. Input `y_pred` is compared with ground truth `y`.
    `y_preds` is expected to have binarized predictions and `y` should be in one-hot format.
    You can use suitable transforms in ``monai.transforms.post`` first to achieve binarized values.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).
    The implementation refers to `DeepMind's implementation <https://github.com/deepmind/surface-distance>`_.

    Args:
        include_background: whether to include distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result. Defaults to ``"mean"``.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        include_background: bool = False,
        distance_metric: str = "euclidean",
        percentile: Optional[float] = None,
        directed: bool = False,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.directed = directed
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute the distance. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")
        if not torch.all(y_pred.byte() == y_pred):
            warnings.warn("y_pred should be a binarized tensor.")
        if not torch.all(y.byte() == y):
            raise ValueError("y should be a binarized tensor.")
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError("y_pred should have at least three dimensions.")
        # compute (BxC) for each channel for each batch
        return compute_hausdorff_distance(
            y_pred=y_pred,
            y=y,
            include_background=self.include_background,
            distance_metric=self.distance_metric,
            percentile=self.percentile,
            directed=self.directed,
        )

    def aggregate(self):  # type: ignore
        """
        Execute reduction logic for the output of `compute_hausdorff_distance`.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_hausdorff_distance(
    y_pred: Union[np.ndarray, torch.Tensor],
    y: Union[np.ndarray, torch.Tensor],
    include_background: bool = False,
    distance_metric: str = "euclidean",
    percentile: Optional[float] = None,
    directed: bool = False,
):
    """
    Compute the Hausdorff distance.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean the distance. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to skip distance computation on the first channel of
            the predicted output. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: whether to calculate directed Hausdorff distance. Defaults to ``False``.
    """

    if not include_background:
        y_pred, y = ignore_background(
            y_pred=y_pred,
            y=y,
        )
    if isinstance(y, torch.Tensor):
        y = y.float()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError("y_pred and y should have same shapes.")

    batch_size, n_class = y_pred.shape[:2]
    hd = np.empty((batch_size, n_class))
    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt) = get_mask_edges(y_pred[b, c], y[b, c])
        if not np.any(edges_gt):
            warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan/inf distance.")
        if not np.any(edges_pred):
            warnings.warn(f"the prediction of class {c} is all 0, this may result in nan/inf distance.")

        distance_1 = compute_percent_hausdorff_distance(edges_pred, edges_gt, distance_metric, percentile)
        if directed:
            hd[b, c] = distance_1
        else:
            distance_2 = compute_percent_hausdorff_distance(edges_gt, edges_pred, distance_metric, percentile)
            hd[b, c] = max(distance_1, distance_2)
    return torch.from_numpy(hd)


def compute_percent_hausdorff_distance(
    edges_pred: np.ndarray,
    edges_gt: np.ndarray,
    distance_metric: str = "euclidean",
    percentile: Optional[float] = None,
):
    """
    This function is used to compute the directed Hausdorff distance.
    """

    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric=distance_metric)

    # for both pred and gt do not have foreground
    if surface_distance.shape == (0,):
        return np.nan

    if not percentile:
        return surface_distance.max()

    if 0 <= percentile <= 100:
        return np.percentile(surface_distance, percentile)
    raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")
