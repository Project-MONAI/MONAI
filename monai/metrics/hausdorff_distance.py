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

from skimage.measure import marching_cubes
from scipy.spatial.distance import directed_hausdorff


def compute_hausdorff_distance(
    seg_1: Union[np.ndarray, torch.Tensor],
    seg_2: Union[np.ndarray, torch.Tensor],
    label_idx: int,
    directed_hausdorff: bool = False
) -> float:
    """
    Compute the directed Hausdorff distance. Here, we consider the directed
    Hausdorff, which is the maximum distance between any point on the first
    set and its nearest point on the second set.

    We require that images are the same size, and assume that they occupy the
    same space (spacing, orientation, etc.).
    """

    # Get both labelfields as np arrays
    if torch.is_tensor(seg_1):
        seg_1 = seg_1.detach().cpu().np()
    if torch.is_tensor(seg_2):
        seg_2 = seg_2.detach().cpu().np()

    # Check non-zero number of elements and same shape
    if seg_1.size == 0 or seg_1.shape != seg_2.shape:
        raise ValueError("Labelfields should have same "
                         "shape (and non-zero number of elements)")

    # Check both have at least 1 voxel with desired index
    if label_idx not in seg_1 or label_idx not in seg_2:
        raise ValueError("Labelfields should have at least 1 voxel containing "
                         f"the desired labelfield, {label_idx}")

    # Generate a surface mesh from each of the images
    coords_1 = marching_cubes(seg_1, label_idx)
    coords_2 = marching_cubes(seg_2, label_idx)

    # TODO Check surfaces
    if len(coords_1)*len(coords_2) <= 1:
        raise ValueError("Marching cubes failed for at least one of the "
                         "labelfields. "
                         f"Num coords for labelfield 1: {len(coords_1)}. "
                         f"Num coords for labelfield 2: {len(coords_2)}.")

    if directed_hausdorff:
        return directed_hausdorff(coords_1, coords_2)
    else:
        return max(
            directed_hausdorff(coords_1, coords_2),
            directed_hausdorff(coords_2, coords_1))
