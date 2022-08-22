import numpy as np

from typing import Any, List, Tuple
from monai.data.meta_tensor import MetaTensor
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

from monai.transforms import CropForeground, ToCupy
from monai.utils import min_version, optional_import

measure_np, has_measure = optional_import("skimage.measure", "0.14.2", min_version)
cp, has_cp = optional_import("cupy")
cucim, has_cucim = optional_import("cucim")

def get_foreground_image(image: MetaTensor):
    """
    Get a foreground image by removing all-zero rectangles on the edges of the image
    Note for the developer: update select_fn if the foreground is defined differently.

    Args:
        image: ndarray image to segment.

    Returns:
        ndarray of foreground image by removing all-zero edges.

    Notes:
        the size of the ouput is smaller than the input.
    """
    # todo(mingxin): type check
    copper = CropForeground(select_fn=lambda x: x > 0)
    image_foreground = copper(image)
    return image_foreground

def get_foreground_label(image: MetaTensor, label: MetaTensor) -> MetaTensor:
    """
    Get foreground image pixel values and mask out the non-labeled area.

    Args
        image: ndarray image to segment.
        label: ndarray the image input and annotated with class IDs.

    Returns:
        1D array of foreground image with label > 0
    """
    # todo(mingxin): type check
    label_foreground = MetaTensor(image[label > 0])
    return label_foreground

def get_label_ccp(mask_index: MetaTensor, use_gpu: bool = True) -> Tuple[List[Any], int]:
    """
    Find all connected components and their bounding shape. Backend can be cuPy/cuCIM or Numpy
    depending on the hardware.

    Args:
        mask_index: a binary mask
        use_gpu: a switch to use GPU/CUDA or not. If GPU is unavailable, CPU will be used
            regardless of this setting

    """
    # todo(mingxin): type check
    shape_list = []
    if mask_index.device.type == "cuda" and has_cp and has_cucim and use_gpu:
        mask_cupy = ToCupy()(mask_index.short())
        labeled = cucim.skimage.measure.label(mask_cupy)
        vals = cp.unique(labeled[cp.nonzero(labeled)])

        for ncomp in vals:
            comp_idx = cp.argwhere(labeled == ncomp)
            comp_idx_min = cp.min(comp_idx, axis=0).tolist()
            comp_idx_max = cp.max(comp_idx, axis=0).tolist()
            bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
            shape_list.append(bbox_shape)
        ncomponents = len(vals)

    elif has_measure:
        labeled, ncomponents = measure_np.label(mask_index.data.cpu().numpy(), background=-1, return_num=True)
        for ncomp in range(1, ncomponents + 1):
            comp_idx = np.argwhere(labeled == ncomp)
            comp_idx_min = np.min(comp_idx, axis=0).tolist()
            comp_idx_max = np.max(comp_idx, axis=0).tolist()
            bbox_shape = [comp_idx_max[i] - comp_idx_min[i] + 1 for i in range(len(comp_idx_max))]
            shape_list.append(bbox_shape)
    else:
        raise RuntimeError("Cannot find one of the following required dependencies: {cuPy+cuCIM} or {scikit-image}")

    return shape_list, ncomponents