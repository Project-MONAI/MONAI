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

import numpy as np
import torch

from .utils import get_mask_edges, get_surface_distance


def compute_hausdorff_distance(
    seg_pred: Union[np.ndarray, torch.Tensor],
    seg_gt: Union[np.ndarray, torch.Tensor],
    label_idx: int,
    distance_metric: str = "euclidean",
    percentile: Optional[float] = None,
    directed: bool = False,
):
    """
    Compute the Hausdorff distance. The user has the option to calculate the
    directed or non-directed Hausdorff distance. By default, the non-directed
    Hausdorff distance is calculated. In addition, specify the `percentile`
    parameter can get the percentile of the distance.

    Args:
        seg_pred: the predicted binary or labelfield image.
        seg_gt: the actual binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
        directed: calculate directed Hausdorff distance. Defaults to ``False``.
    """

    (edges_pred, edges_gt) = get_mask_edges(seg_pred, seg_gt, label_idx)
    hd = compute_percent_hausdorff_distance(edges_pred, edges_gt, label_idx, distance_metric, percentile)
    if directed:
        return hd

    hd2 = compute_percent_hausdorff_distance(edges_gt, edges_pred, label_idx, distance_metric, percentile)
    return max(hd, hd2)


def compute_percent_hausdorff_distance(
    edges_pred: np.ndarray,
    edges_gt: np.ndarray,
    label_idx: int,
    distance_metric: str = "euclidean",
    percentile: Optional[float] = None,
):
    """
    This function is used to compute the directed Hausdorff distance.

    Args:
        edges_pred: the edge of the predictions.
        edges_gt: the edge of the ground truth.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
        percentile: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
    """

    surface_distance = get_surface_distance(edges_pred, edges_gt, label_idx, distance_metric=distance_metric)

    # for input without foreground
    if surface_distance.shape == (0,):
        return np.inf

    if not percentile:
        return surface_distance.max()
    elif 0 <= percentile <= 100:
        return np.percentile(surface_distance, percentile)
    else:
        raise ValueError(f"percentile should be a value between 0 and 100, get {percentile}.")
