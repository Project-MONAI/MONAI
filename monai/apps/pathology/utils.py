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

from typing import List, Union

import numpy as np
import torch

from monai.transforms.post.array import ProbNMS
from monai.utils import optional_import

measure, _ = optional_import("skimage.measure")
ndimage, _ = optional_import("scipy.ndimage")


def compute_multi_instance_mask(mask: np.ndarray, threshold: float):
    """
    This method computes the segmentation mask according to the binary tumor mask.

    Args:
        mask: the binary mask array
        threshold: the threshold to fill holes
    """

    neg = 255 - mask * 255
    distance = ndimage.morphology.distance_transform_edt(neg)
    binary = distance < threshold

    filled_image = ndimage.morphology.binary_fill_holes(binary)
    multi_instance_mask = measure.label(filled_image, connectivity=2)

    return multi_instance_mask


def compute_isolated_tumor_cells(tumor_mask: np.ndarray, threshold: float) -> List[int]:
    """
    This method computes identifies Isolated Tumor Cells (ITC) and return their labels.

    Args:
        tumor_mask: the tumor mask.
        threshold: the threshold (at the mask level) to define an isolated tumor cell (ITC).
            A region with the longest diameter less than this threshold is considered as an ITC.
    """
    max_label = np.amax(tumor_mask)
    properties = measure.regionprops(tumor_mask, coordinates="rc")
    itc_list = [i + 1 for i in range(max_label) if properties[i].major_axis_length < threshold]

    return itc_list


class PathologyProbNMS(ProbNMS):
    """
    This class extends monai.utils.ProbNMS and add the `resolution` option for
    Pathology.
    """

    def __call__(self, probs_map: Union[np.ndarray, torch.Tensor], resolution_level: int = 0):
        """
        probs_map: the input probabilities map, it must have shape (H[, W, ...]).
        resolution_level: the level at which the probabilities map is made.
        """
        resolution = pow(2, resolution_level)
        org_outputs = ProbNMS.__call__(self, probs_map)
        outputs = []
        for org_output in org_outputs:
            prob = org_output[0]
            coord = np.asarray(org_output[1:])
            coord_wsi = ((coord + 0.5) * resolution).astype(int)
            outputs.append([prob] + list(coord_wsi))
        return outputs
