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

from typing import Union

import numpy as np
import torch

from .utils import get_mask_edges, get_surface_distance


def compute_average_surface_distance(
    seg_pred: Union[np.ndarray, torch.Tensor],
    seg_gt: Union[np.ndarray, torch.Tensor],
    label_idx: int,
    symmetric: bool = False,
    distance_metric: str = "euclidean",
):
    """
    This function is used to compute the Average Surface Distance from `seg_pred` to `seg_gt`
    under the default setting.
    In addition, if sets ``symmetric = True``, the average symmetric surface distance between
    these two inputs will be returned.

    Args:
        seg_pred: first binary or labelfield image.
        seg_gt: second binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_pred = seg_pred == label_idx`.
        symmetric: if calculate the symmetric average surface distance between
            `seg_pred` and `seg_gt`. Defaults to ``False``.
        distance_metric: : [``"euclidean"``, ``"chessboard"``, ``"taxicab"``]
            the metric used to compute surface distance. Defaults to ``"euclidean"``.
    """
    (edges_pred, edges_gt) = get_mask_edges(seg_pred, seg_gt, label_idx)
    surface_distance = get_surface_distance(edges_pred, edges_gt, label_idx, distance_metric=distance_metric)
    if surface_distance.shape == (0,):
        return np.inf

    avg_surface_distance = surface_distance.mean()
    if not symmetric:
        return avg_surface_distance

    surface_distance_2 = get_surface_distance(edges_gt, edges_pred, label_idx, distance_metric=distance_metric)
    if surface_distance_2.shape == (0,):
        return np.inf

    avg_surface_distance_2 = surface_distance_2.mean()
    return np.mean((avg_surface_distance, avg_surface_distance_2))
