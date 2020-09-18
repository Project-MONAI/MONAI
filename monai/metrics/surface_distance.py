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

from typing import Optional, Tuple, Union

import numpy as np
import torch

from monai.transforms.croppad.array import SpatialCrop
from monai.transforms.utils import generate_spatial_bounding_box
from monai.utils import optional_import

binary_erosion, _ = optional_import("scipy.ndimage.morphology", name="binary_erosion")
distance_transform_edt, _ = optional_import("scipy.ndimage.morphology", name="distance_transform_edt")


def get_mask_edges(
    seg_1: Union[np.ndarray, torch.Tensor],
    seg_2: Union[np.ndarray, torch.Tensor],
    label_idx: int,
    crop: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Do binary erosion and use XOR for input to get the edges. This
    function is helpful to further calculate metrics such as Average Surface
    Distance and Hausdorff Distance.
    The input images can be binary or labelfield images. If labelfield images
    are supplied, they are converted to binary images using `label_idx`.

    `scipy`'s Binary erosion is used to to calculate the edges of the binary
    labelfield, and the coordinates of these edges are passed to `scipy`'s
    `directed_hausdorff` function.

    In order to improve the computing efficiency, before getting the edges,
    the images can be cropped and only keep the foreground if not specifies
    ``crop = False``.

    We require that images are the same size, and assume that they occupy the
    same space (spacing, orientation, etc.).

    Args:
        seg_1: first binary or labelfield image.
        seg_2: second binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_1 = seg_1 == label_idx`.
        crop: crop input images and only keep the foregrounds. In order to
            maintain two inputs' shapes, here the bounding box is achieved
            by ``(seg_1 | seg_2)`` which represents the union set of two
            images. Defaults to ``True``.
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

    if crop:
        seg_1, seg_2 = np.expand_dims(seg_1, 0), np.expand_dims(seg_2, 0)
        box_start, box_end = generate_spatial_bounding_box(seg_1 | seg_2)
        cropper = SpatialCrop(roi_start=box_start, roi_end=box_end)
        seg_1, seg_2 = np.squeeze(cropper(seg_1)), np.squeeze(cropper(seg_2))

    # Do binary erosion and use XOR to get edges
    edges_1 = binary_erosion(seg_1) ^ seg_1
    edges_2 = binary_erosion(seg_2) ^ seg_2

    return (edges_1, edges_2)


def get_surface_distance(
    seg_1: Union[np.ndarray, torch.Tensor],
    seg_2: Union[np.ndarray, torch.Tensor],
    label_idx: int,
    crop: bool = True,
) -> np.ndarray:
    """
    This function is used to compute the surface distances from `seg_1` to `seg_2`.

    In order to improve the computing efficiency, before getting the edges,
    the images can be cropped and only keep the foreground if not specifies
    ``crop = False``.

    Args:
        seg_1: first binary or labelfield image.
        seg_2: second binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_1 = seg_1 == label_idx`.
        crop: crop input images and only keep the foregrounds. In order to
            maintain two inputs' shapes, here the bounding box is achieved
            by ``(seg_1 | seg_2)`` which represents the union set of two
            images. Defaults to ``True``.
    """
    (edges_1, edges_2) = get_mask_edges(seg_1, seg_2, label_idx, crop)
    dis_1 = distance_transform_edt(~edges_2)
    surface_distance_1 = dis_1[edges_1]
    return surface_distance_1


def compute_percent_hausdorff_distance(
    surface_distance_1: np.ndarray,
    surface_distance_2: np.ndarray,
    percent: Optional[float] = None,
):
    """
    This function is used to compute the maximum Hausdorff Distance. Specify the `percent`
    parameter can get the percentile of the distance. The surface distance inputs can get
    from ``get_surface_distance`` function. For example:

    .. code-block:: python
        surface_distance_1 = get_surface_distance(seg_1, seg_2, 1)
        surface_distance_2 = get_surface_distance(seg_2, seg_1, 1)

    Args:
        surface_distance_1: the surface distances from the first mask to the second mask.
        surface_distance_2: the surface distances from the second mask to the first mask.
        percent: an optional float number between 0 and 100. If specified, the corresponding
            percentile of the Hausdorff Distance rather than the maximum result will be achieved.
            Defaults to ``None``.
    """
    if not percent:
        hausdorff_distance = max(surface_distance_1.max(), surface_distance_2.max())
        return hausdorff_distance
    elif not percent >= 0 and percent <= 100:
        raise ValueError(f"percent should be a value between 0 and 100, get {percent}")
    else:
        all_dis = np.hstack((surface_distance_1, surface_distance_2))
        per_hausdorff_distance = np.percentile(all_dis, percent)
        return per_hausdorff_distance


def compute_average_surface_distance(
    seg_1: Union[np.ndarray, torch.Tensor],
    seg_2: Union[np.ndarray, torch.Tensor],
    label_idx: int,
    symmetric: bool = False,
):
    """
    This function is used to compute the Average Surface Distance from `seg_1` to `seg_2`
    under the default seeting.
    In addition, if sets ``symmetric = True``, the average symmetric surface distance between
    these two inputs will be returned.

    Args:
        seg_1: first binary or labelfield image.
        seg_2: second binary or labelfield image.
        label_idx: for labelfield images, convert to binary with
            `seg_1 = seg_1 == label_idx`.
        symmetric: if calculate the symmetric average surface distance between
            `seg_1` and `seg_2`. Defaults to ``False``.
    """
    avg_surface_distance_1 = get_surface_distance(seg_1, seg_2, label_idx).mean()
    if not symmetric:
        return avg_surface_distance_1
    else:
        avg_surface_distance_2 = get_surface_distance(seg_2, seg_1, label_idx).mean()
        return np.mean((avg_surface_distance_1, avg_surface_distance_2))
