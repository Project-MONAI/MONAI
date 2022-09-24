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

from typing import Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.post.array import AsDiscrete, FillHoles, RemoveSmallObjects
from monai.transforms.transform import Transform
from monai.utils import TransformBackends, convert_to_numpy, convert_to_tensor, optional_import
from monai.utils.type_conversion import convert_to_dst_type

label, _ = optional_import("scipy.ndimage.measurements", name="label")
disk, _ = optional_import("skimage.morphology", name="disk")
opening, _ = optional_import("skimage.morphology", name="opening")
watershed, _ = optional_import("skimage.segmentation", name="watershed")
gaussian, _ = optional_import("skimage.filters", name="gaussian")
cv2, _ = optional_import("cv2")

__all__ = ["CalcualteInstanceSegmentationMap"]


class CalcualteInstanceSegmentationMap(Transform):
    """
    Process Nuclei Prediction with XY Coordinate Map.

    Args:
        threshold_pred: threshold the float values of prediction to int 0 or 1 with specified theashold. Defaults to 0.5.
        threshold_overall: threshold the float values of overall gradient map to int 0 or 1 with specified theashold.
            Defaults to 0.4.
        min_size: objects smaller than this size are removed. Defaults to 10.
        sigma: std. could be a single value, or `spatial_dims` number of values. Defaults to 0.4.
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        radius: the radius of the disk-shaped footprint. Defaults to 2.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        threshold_pred: float = 0.5,
        threshold_overall: float = 0.4,
        min_size: int = 10,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.4,
        kernel_size: int = 21,
        radius: int = 2,
    ) -> None:
        self.sigma = sigma
        self.radius = radius
        self.kernel_size = kernel_size

        self.remove_small_objects = RemoveSmallObjects(min_size=min_size)
        self.pred_discreter = AsDiscrete(threshold=threshold_pred)
        self.overall_discreter = AsDiscrete(threshold=threshold_overall)
        self.fill_holes = FillHoles()

    def __call__(self, prob_map: NdarrayOrTensor, hover_map: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            prob_map: the probability map output of the NP branch, shape must be [1, H, W].
            hover_map: the horizontal and vertical distances of nuclear pixels to their centres
                of mass output from the HV branch, shape must be [2, H, W].

        Returns:
            instance labelled segmentation map with shape [1, H, W].
        """
        prob_map = convert_to_tensor(prob_map, track_meta=get_track_meta())
        hover_map = convert_to_tensor(hover_map, track_meta=get_track_meta())

        pred = convert_to_numpy(prob_map)
        hover_map = convert_to_numpy(hover_map)

        # processing
        pred = self.pred_discreter(pred)
        pred = label(pred)[0]
        pred = self.remove_small_objects(pred)
        pred[pred > 0] = 1

        hover_h = hover_map[0, ...]
        hover_v = hover_map[1, ...]

        hover_h = (hover_h - np.min(hover_h)) / (np.max(hover_h) - np.min(hover_h))
        hover_v = (hover_v - np.min(hover_v)) / (np.max(hover_v) - np.min(hover_v))
        sobelh = cv2.Sobel(hover_h, cv2.CV_64F, 1, 0, ksize=self.kernel_size)
        sobelv = cv2.Sobel(hover_v, cv2.CV_64F, 0, 1, ksize=self.kernel_size)
        sobelh = 1 - (sobelh - np.min(sobelh)) / (np.max(sobelh) - np.min(sobelh))
        sobelv = 1 - (sobelv - np.min(sobelv)) / (np.max(sobelv) - np.min(sobelv))

        # combine the h & v values using max
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - pred)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * pred

        # nuclei values form mountains so inverse to get basins
        dist = -gaussian(dist, sigma=self.sigma)

        overall = self.overall_discreter(overall)

        marker = pred - overall
        marker[marker < 0] = 0
        marker = self.fill_holes(marker)

        marker = opening(marker.squeeze(0), disk(self.radius))
        marker = label(marker)[0]
        marker = self.remove_small_objects(marker[None])

        proced_pred = watershed(dist, markers=marker, mask=pred)
        pred, *_ = convert_to_dst_type(proced_pred, prob_map)

        return pred  # type: ignore
