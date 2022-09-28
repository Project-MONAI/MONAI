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

from dis import dis
from typing import Callable, Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.post.array import AsDiscrete, FillHoles, RemoveSmallObjects, SobelGradients
from monai.transforms.transform import Transform
from monai.utils import TransformBackends, convert_to_numpy, optional_import
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor

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
        threshold_overall: threshold the float values of overall gradient map to int 0 or 1 with specified theashold.
            Defaults to 0.4.
        min_size: objects smaller than this size are removed. Defaults to 10.
        sigma: std. Used in `GaussianFilter`. Could be a single value, or `spatial_dims` number of values. Defaults to 0.4.
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        radius: the radius of the disk-shaped footprint. Defaults to 2.
        gaussian: whether need to smooth the image to be applied by the watershed segmentation. Defaults to False.
        remove_small_objects: whether need to remove some objects in segmentation results and marker. Defaults to False.
        marker_postprocess_fn: execute additional post transformation on marker. Defaults to None.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        threshold_overall: float = 0.4,
        min_size: int = 10,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.4,
        kernel_size: int = 21,
        radius: int = 2,
        gaussian: bool = False,
        remove_small_objects: bool = False,
        marker_postprocess_fn: Callable = None,
    ) -> None:
        self.sigma = sigma
        self.radius = radius
        self.kernel_size = kernel_size
        self.threshold_overall = threshold_overall
        self.gaussian = gaussian
        self.remove_small_objects = remove_small_objects
        self.marker_postprocess_fn = marker_postprocess_fn

        if self.remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)

    def __call__(self, seg_pred: NdarrayOrTensor, hover_map: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            seg_pred: the output of the NP(segmentation) branch, shape must be [1, H, W].
                `seg_pred` have been applied activation layer and must be binarized.
            hover_map: the horizontal and vertical distances of nuclear pixels to their centres
                of mass output from the HV branch, shape must be [2, H, W].

        Returns:
            instance labelled segmentation map with shape [1, H, W].

        Raises:
            ValueError: when the `seg_pred` dimension is not [1, H, W].
            ValueError: when the `hover_map` dimension is not [2, H, W].

        """
        if len(seg_pred.shape) != 3 or len(hover_map.shape) != 3:  # only for 2D
            raise ValueError("Only support 2D, shape must be [C, H, W]!")
        if seg_pred.shape[0] != 1:
            raise ValueError("Only supports single channel segmentation prediction!")
        if hover_map.shape[0] != 2:
            raise ValueError("Hover map should be with shape [2, H, W]!")

        if isinstance(seg_pred, torch.Tensor):
            seg_pred = seg_pred.detach().cpu().numpy()
        pred = convert_to_tensor(seg_pred, track_meta=get_track_meta())
        hover_map = convert_to_tensor(hover_map, track_meta=get_track_meta())

        pred = label(pred)[0]
        if self.remove_small_objects:
            pred = self.remove_small_objects(pred)
        pred[pred > 0] = 1

        hover_h = hover_map[0: 1, ...]
        hover_v = hover_map[1: 2, ...]

        hover_h = (hover_h - np.min(hover_h)) / (np.max(hover_h) - np.min(hover_h))
        hover_v = (hover_v - np.min(hover_v)) / (np.max(hover_v) - np.min(hover_v))
        sobelh = SobelGradients(kernel_size=self.kernel_size)(hover_h)[0, ...]
        sobelv = SobelGradients(kernel_size=self.kernel_size)(hover_v)[1, ...]
        sobelh = 1 - (sobelh - np.min(sobelh)) / (np.max(sobelh) - np.min(sobelh))
        sobelv = 1 - (sobelv - np.min(sobelv)) / (np.max(sobelv) - np.min(sobelv))

        # combine the h & v values using max
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - pred)
        overall[overall < 0] = 0

        dist = (1.0 - overall) * pred
        if self.gaussian:
            spatial_dim = len(dist.shape)- 1
            gaussian = GaussianFilter(spatial_dims=spatial_dim, sigma=self.sigma)
            dist = convert_to_tensor(dist[None])
            dist = convert_to_numpy(gaussian(dist)).squeeze(0)

        overall = overall >= self.threshold_overall

        marker = pred - overall
        marker[marker < 0] = 0
        if self.marker_postprocess_fn:
            marker = self.marker_postprocess_fn(marker)
        marker = opening(marker.squeeze(), disk(self.radius))
        marker = label(marker)[0]
        if self.remove_small_objects:
            marker = self.remove_small_objects(marker[None])

        # nuclei values form mountains so inverse to get basins
        proced_pred = watershed(-dist, markers=marker, mask=pred)
        pred, *_ = convert_to_dst_type(proced_pred, seg_pred)

        return pred  # type: ignore
