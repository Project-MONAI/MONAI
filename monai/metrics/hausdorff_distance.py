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

from monai.utils import optional_import

binary_dilation, _ = optional_import("scipy.ndimage.morphology", name="binary_dilation")
directed_hausdorff, _ = optional_import("scipy.spatial.distance", name="directed_hausdorff")


def compute_hausdorff_distance(
    seg_1: Union[np.ndarray, torch.Tensor],
    seg_2: Union[np.ndarray, torch.Tensor],
    label_idx: int,
    directed: bool = False,
) -> float:
    """
    Compute the Hausdorff distance. The user has the option to calculate the
    directed or non-directed Hausdorff distance. By default, the non-directed
    Hausdorff distance is calculated.

    The input images can be binary or labelfield images. If labelfield images
    are supplied, they are converted to binary images using `label_idx`.

    `scipy`'s Binary dilation is used to to calculate the edges of the binary
    labelfield, and the coordinates of these edges are passed to `scipy`'s
    `directed_hausdorff` function.

    We require that images are the same size, and assume that they occupy the
    same space (spacing, orientation, etc.).

    Args:
        seg_1: first binary or labelfield image.
        seg_2: second binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_1 = seg_1 == label_idx`.
        directed: calculate directed Hausdorff distance (defaults to `False`).
    """

    # Get both labelfields as np arrays
    if torch.is_tensor(seg_1):
        seg_1 = seg_1.detach().cpu().numpy()
    if torch.is_tensor(seg_2):
        seg_2 = seg_2.detach().cpu().numpy()

    # Check non-zero number of elements and same shape
    if seg_1.size == 0 or seg_1.shape != seg_2.shape:
        raise ValueError("Labelfields should have same shape (and non-zero number of elements)")

    # If not binary images, convert them
    if seg_1.dtype != bool:
        seg_1 = seg_1 == label_idx
    if seg_2.dtype != bool:
        seg_2 = seg_2 == label_idx

    # Check both have at least 1 voxel with desired index
    if not (np.any(seg_1) and np.any(seg_2)):
        raise ValueError(f"Labelfields should have at least 1 voxel containing the desired labelfield, {label_idx}")

    # Do binary dilation and use XOR to get edges
    edges_1 = binary_dilation(seg_1) ^ seg_1
    edges_2 = binary_dilation(seg_2) ^ seg_2

    # Extract coordinates of these edge points
    coords_1 = np.argwhere(edges_1)
    coords_2 = np.argwhere(edges_2)

    # Get (potentially directed) Hausdorff distance
    if directed:
        return float(directed_hausdorff(coords_1, coords_2)[0])
    else:
        return float(max(directed_hausdorff(coords_1, coords_2)[0], directed_hausdorff(coords_2, coords_1)[0]))
