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

from typing import Callable, Sequence, Union, Optional

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms.post.array import Activations, AsDiscrete, RemoveSmallObjects, SobelGradients
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.transform import Transform
from monai.utils import TransformBackends, convert_to_numpy, optional_import
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor

label, _ = optional_import("scipy.ndimage.measurements", name="label")
disk, _ = optional_import("skimage.morphology", name="disk")
opening, _ = optional_import("skimage.morphology", name="opening")
watershed, _ = optional_import("skimage.segmentation", name="watershed")

__all__ = ["CalculateInstanceSegmentationMap", "GenerateMask", "GenerateProbabilityMap", "GenerateDistanceMap", "GenerateMarkers"]


class CalculateInstanceSegmentationMap(Transform):
    """
    Use `skimage.segmentation.watershed` to get instance segmentation results from images.
    See: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed.

    Args:
        connectivity: An array with the same number of dimensions as image whose non-zero elements indicate neighbors for connection.
            Following the scipy convention, default is a one-connected array of the dimension of the image.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, connectivity: Optional[int] = 1) -> None:
        self.connectivity = connectivity

    def __call__(self, image: NdarrayOrTensor, mask: NdarrayOrTensor, markers: Optional[NdarrayOrTensor] = None) -> NdarrayOrTensor:
        """
        Args:
            image: image where the lowest value points are labeled first. Shape must be [1, H, W].
            mask: the same shape as image. Only points at which mask == True will be labeled.
            markers: The desired number of markers, or an array marking the basins with the values to be assigned in the label matrix.
                Zero means not a marker. If None (no markers given), the local minima of the image are used as markers. Shape must be [1, H, W].
        """

        image = convert_to_numpy(image)
        markers = convert_to_numpy(markers)
        mask = convert_to_numpy(mask)

        instance_seg = watershed(image, markers=markers, mask=mask, connectivity=self.connectivity)

        return convert_to_dst_type(instance_seg, image, dtype=np.uint8)[0]


class GenerateMask(Transform):
    """
    generate mask used in `watershed`. Only points at which mask == True will be labeled.

    Args:
        softmax: if True, apply a softmax function to the prediction.
        sigmoid: if True, apply a sigmoid function to the prediction.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        softmax: bool = True,
        sigmoid: bool = False,
        threshold: Optional[float] = None,
        remove_small_objects: bool = True,
        min_size: int = 10
    ) -> None:
        if sigmoid and threshold is None:
            raise ValueError("Threshold is needed when using sigmoid activation.")

        self.activations = Activations(sigmoid=sigmoid, softmax=softmax)
        self.asdiscrete = AsDiscrete(threshold=threshold, argmax=softmax)
        if remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)
        else:
            self.remove_small_objects = None  # type: ignore

    def __call__(self, prob_map: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            prob_map: probability map of segmentation, shape must be [C, H, W]
        """
        if len(prob_map.shape) != 3:
            raise ValueError(f"Suppose the input probability map should be with shape of [C, H, W], but got {prob_map.shape}")

        pred = self.activations(prob_map)
        pred = self.asdiscrete(pred)

        if isinstance(pred, torch.Tensor):
            pred = pred.cpu()

        pred = label(pred)[0]
        if self.remove_small_objects:
            pred = self.remove_small_objects(pred)
        pred[pred > 0] = 1

        return convert_to_dst_type(pred, prob_map, dtype=np.uint8)[0]


class GenerateProbabilityMap(Transform):
    """
    Generate foreground probability map by hover map. The more parts of the image that cannot be identified as foreground areas,
    the larger the grey scale value. The grey value of the instance's border will be larger.

    Args:
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        remove_small_objects: whether need to remove some objects in segmentation results. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        kernel_size: int = 21,
        min_size: int = 10,
        remove_small_objects: bool = True,
    ) -> None:
        self.sobel_gradient =  SobelGradients(kernel_size=kernel_size)
        if remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)
        else:
            self.remove_small_objects = None  # type: ignore

    def __call__(self, mask: NdarrayOrTensor, hover_map: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            mask: binarized segmentation result.  Shape must be [1, H, W].
            hover_map:  horizontal and vertical distances of nuclear pixels to their centres of mass. Shape must be [2, H, W].
                The first and second channel represent the horizontal and vertical maps respectively. For more details refer
                to papers: https://arxiv.org/abs/1812.06499.

        Return:
            Foreground probability map.
        """
        mask = convert_to_tensor(mask, track_meta=get_track_meta())
        hover_map = convert_to_tensor(hover_map, track_meta=get_track_meta())

        hover_h = hover_map[0:1, ...]
        hover_v = hover_map[1:2, ...]

        hover_h = (hover_h - np.min(hover_h)) / (np.max(hover_h) - np.min(hover_h))
        hover_v = (hover_v - np.min(hover_v)) / (np.max(hover_v) - np.min(hover_v))
        sobelh = self.sobel_gradient(hover_h)[0, ...]
        sobelv = self.sobel_gradient(hover_v)[1, ...]
        sobelh = 1 - (sobelh - np.min(sobelh)) / (np.max(sobelh) - np.min(sobelh))
        sobelv = 1 - (sobelv - np.min(sobelv)) / (np.max(sobelv) - np.min(sobelv))

        # combine the h & v values using max
        overall = np.maximum(sobelh, sobelv)
        overall = overall - (1 - mask)
        overall[overall < 0] = 0

        return convert_to_dst_type(overall, mask, dtype=np.float32)[0]


class GenerateDistanceMap(Transform):
    """
    Generate distance map.
    In general, the instance map is calculated from the distance to the background. Here, we use 1 - "foreground probability map" to
    generate the distance map. Nnuclei values form mountains so inverse to get basins.

    Args:
        sigma: std. Used in `GaussianSmooth`. Could be a single value, or `spatial_dims` number of values. Defaults to 0.4.
        smooth_fn: execute smooth function on distance map. Defaults to None. You can specify "guassian" or other callable functions.
            If specify "guassian", `GaussianSmooth` will applied on distance map.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        sigma: Union[Sequence[float], float] = 0.4,
        smooth_fn: Optional[Union[Callable, str]] = None,
    ) -> None:
        self.sigma = sigma
        self.smooth_fn = smooth_fn

    def __call__(self, mask: NdarrayOrTensor, foreground_prob_map: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            mask: binarized segmentation result. Shape must be [1, H, W].
            foreground_prob_map: foreground probability map. Shape must be [1, H, W].
        """
        mask = convert_to_tensor(mask, track_meta=get_track_meta())
        foreground_prob_map = convert_to_tensor(foreground_prob_map, track_meta=get_track_meta())

        distance_map = (1.0 - foreground_prob_map) * mask

        if self.smooth_fn == "gaussian":
            gaussian = GaussianSmooth(sigma=self.sigma)
            distance_map = gaussian(distance_map)
        elif callable(self.smooth_fn):
            distance_map = self.smooth_fn(distance_map)

        return convert_to_dst_type(-distance_map, mask, dtype=np.float32)[0]


class GenerateMarkers(Transform):
    """
    Generate markers used in `watershed`. Generally, The maximum of this distance (i.e., the minimum of the opposite of the distance)
    are chosen as markers and the flooding of basins from such markers separates the two instances along a watershed line.
    Here is the implementation from HoVerNet papar. For more details refer to papers: https://arxiv.org/abs/1812.06499.

    Args:
        threshold: threshold the float values of foreground probability map to int 0 or 1 with specified theashold.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        postprocess_fn: execute additional post transformation on marker. Defaults to None.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        threshold: float = 0.4,
        radius: int = 2,
        min_size: int = 10,
        remove_small_objects: bool = True,
        postprocess_fn: Callable = None,
    ) -> None:
        self.threshold = threshold
        self.radius = radius
        self.postprocess_fn = postprocess_fn

        if remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)

    def __call__(self, mask: NdarrayOrTensor, foreground_prob_map: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            mask: binarized segmentation result. Shape must be [1, H, W].
            foreground_prob_map: foreground probability map. Shape must be [1, H, W].
        """
        mask = convert_to_tensor(mask, track_meta=get_track_meta())
        foreground_prob_map = convert_to_tensor(foreground_prob_map, track_meta=get_track_meta())

        foreground_prob_map = foreground_prob_map >= self.threshold  # uncertain area

        marker = mask - foreground_prob_map.astype(torch.uint8) # certain foreground
        marker[marker < 0] = 0
        if self.postprocess_fn:
            marker = self.postprocess_fn(marker)

        marker = opening(marker.squeeze(), disk(self.radius))
        marker = label(marker)[0]
        if self.remove_small_objects:
            marker = self.remove_small_objects(marker[None])

        return convert_to_dst_type(marker, mask, dtype=np.uint8)[0]
