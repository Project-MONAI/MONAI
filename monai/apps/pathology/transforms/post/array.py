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

from typing import Callable, Optional

import numpy as np
import torch

from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.transforms.post.array import Activations, AsDiscrete, RemoveSmallObjects, SobelGradients
from monai.transforms.transform import Transform
from monai.transforms.utils_pytorch_numpy_unification import max, maximum, min
from monai.utils import TransformBackends, convert_to_numpy, optional_import
from monai.utils.type_conversion import convert_to_dst_type

label, _ = optional_import("scipy.ndimage.measurements", name="label")
disk, _ = optional_import("skimage.morphology", name="disk")
opening, _ = optional_import("skimage.morphology", name="opening")
watershed, _ = optional_import("skimage.segmentation", name="watershed")

__all__ = [
    "CalculateInstanceSegmentationMap",
    "GenerateMask",
    "GenerateProbabilityMap",
    "GenerateDistanceMap",
    "GenerateMarkers",
]


class CalculateInstanceSegmentationMap(Transform):
    """
    Use `skimage.segmentation.watershed` to get instance segmentation results from images.
    See: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed.

    Args:
        connectivity: An array with the same number of dimensions as image whose non-zero elements indicate
            neighbors for connection. Following the scipy convention, default is a one-connected array of
            the dimension of the image.
        dtype: target data content type to convert, default is np.uint8.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, connectivity: Optional[int] = 1, dtype: DtypeLike = np.uint8) -> None:
        self.connectivity = connectivity
        self.dtype = dtype

    def __call__(  # type: ignore
        self, image: NdarrayOrTensor, mask: Optional[NdarrayOrTensor] = None, markers: Optional[NdarrayOrTensor] = None
    ) -> NdarrayOrTensor:
        """
        Args:
            image: image where the lowest value points are labeled first. Shape must be [1, H, W].
            mask: optional, the same shape as image. Only points at which mask == True will be labeled.
            markers: The desired number of markers, or an array marking the basins with the values to be assigned
                in the label matrix. Zero means not a marker. If None (no markers given), the local minima of the
                image are used as markers. Shape must be [1, H, W].
        """

        image = convert_to_numpy(image)
        markers = convert_to_numpy(markers)
        mask = convert_to_numpy(mask)

        if image.shape[0] != 1 or image.ndim != 3:
            raise ValueError(f"Input image should be with size of [1, H, W], but got {image.shape}")
        if mask is not None and (mask.shape[0] != 1 or mask.ndim != 3):
            raise ValueError(f"Input mask should be with size of [1, H, W], but got {mask.shape}")
        if markers is not None and (markers.shape[0] != 1 or markers.ndim != 3):
            raise ValueError(f"Input markers should be with size of [1, H, W], but got {markers.shape}")

        instance_seg = watershed(image, markers=markers, mask=mask, connectivity=self.connectivity)

        return convert_to_dst_type(instance_seg, image, dtype=self.dtype)[0]


class GenerateMask(Transform):
    """
    generate mask used in `watershed`. Only points at which mask == True will be labeled.

    Args:
        softmax: if True, apply a softmax function to the prediction.
        sigmoid: if True, apply a sigmoid function to the prediction.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        dtype: target data content type to convert, default is np.uint8.

    Raises:
        ValueError: when the `prob_map` shape is not [C, H, W].

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        softmax: bool = True,
        sigmoid: bool = False,
        threshold: Optional[float] = None,
        remove_small_objects: bool = True,
        min_size: int = 10,
        dtype: DtypeLike = np.uint8,
    ) -> None:
        if sigmoid and threshold is None:
            raise ValueError("Threshold is needed when using sigmoid activation.")

        self.dtype = dtype
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
            raise ValueError(
                f"Suppose the input probability map should be with shape of [C, H, W], but got {prob_map.shape}"
            )

        pred = self.activations(prob_map)
        pred = self.asdiscrete(pred)

        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu()

        pred = label(pred)[0]
        if self.remove_small_objects:
            pred = self.remove_small_objects(pred)
        pred[pred > 0] = 1  # type: ignore

        return convert_to_dst_type(pred, prob_map, dtype=self.dtype)[0]


class GenerateProbabilityMap(Transform):
    """
    Generate foreground probability map by hover map. The more parts of the image that cannot be identified as foreground areas,
    the larger the grey scale value. The grey value of the instance's border will be larger.

    Args:
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        remove_small_objects: whether need to remove some objects in segmentation results. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        dtype: target data content type to convert, default is np.float32.


    Raises:
        ValueError: when the `mask` shape is not [1, H, W].
        ValueError: when the `hover_map` shape is not [2, H, W].

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self, 
        kernel_size: int = 21, 
        min_size: int = 10, 
        remove_small_objects: bool = True,
        dtype : DtypeLike = np.float32
    ) -> None:

        self.dtype = dtype

        self.sobel_gradient = SobelGradients(kernel_size=kernel_size)
        if remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)
        else:
            self.remove_small_objects = None  # type: ignore

    def __call__(self, mask: NdarrayOrTensor, hover_map: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binarized segmentation result.  Shape must be [1, H, W].
            hover_map:  horizontal and vertical distances of nuclear pixels to their centres of mass. Shape must be [2, H, W].
                The first and second channel represent the horizontal and vertical maps respectively. For more details refer
                to papers: https://arxiv.org/abs/1812.06499.

        Return:
            Foreground probability map.
        
        Raises:
            ValueError: when the `hover_map` has only one value.
            ValueError: when the `sobel gradient map` has only one value.

        """
        if len(mask.shape) != 3 or len(hover_map.shape) != 3:
            raise ValueError(
                f"Suppose the mask and hover map should be with shape of [C, H, W], but got {mask.shape}, {hover_map.shape}"
            )
        if mask.shape[0] != 1:
            raise ValueError(f"Suppose the mask only has one channel, but got {mask.shape[0]}")
        if hover_map.shape[0] != 2:
            raise ValueError(f"Suppose the hover map only has two channels, but got {hover_map.shape[0]}")

        hover_h = hover_map[0:1, ...]
        hover_v = hover_map[1:2, ...]

        if (max(hover_h) - min(hover_h)) == 0 or (max(hover_v) - min(hover_v)) == 0:
            raise ValueError("Not a valid hover map, please check your input")
        hover_h = (hover_h - min(hover_h)) / (max(hover_h) - min(hover_h))
        hover_v = (hover_v - min(hover_v)) / (max(hover_v) - min(hover_v))
        sobelh = self.sobel_gradient(hover_h)[0, ...]
        sobelv = self.sobel_gradient(hover_v)[1, ...]
        if (max(sobelh) - min(sobelh)) == 0 or (max(sobelv) - min(sobelv)) == 0:
            raise ValueError("Not a valid sobel gradient map")
        sobelh = 1 - (sobelh - min(sobelh)) / (max(sobelh) - min(sobelh))
        sobelv = 1 - (sobelv - min(sobelv)) / (max(sobelv) - min(sobelv))

        # combine the h & v values using max
        overall = maximum(sobelh, sobelv)
        overall = overall - (1 - mask)
        overall[overall < 0] = 0

        return convert_to_dst_type(overall, mask, dtype=self.dtype)[0]


class GenerateDistanceMap(Transform):
    """
    Generate distance map.
    In general, the instance map is calculated from the distance to the background.
    Here, we use 1 - "foreground probability map" to generate the distance map.
    Nuclei values form mountains so inverse to get basins.

    Args:
        smooth_fn: execute smooth function on distance map. Defaults to None. You can specify
            callable functions for smoothing.
            For example, if you want apply gaussian smooth, you can specify `smooth_fn = GaussianSmooth()`
        dtype: target data content type to convert, default is np.float32.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, smooth_fn: Optional[Callable] = None, dtype: DtypeLike = np.float32) -> None:
        self.smooth_fn = smooth_fn
        self.dtype = dtype

    def __call__(self, mask: NdarrayOrTensor, foreground_prob_map: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binarized segmentation result. Shape must be [1, H, W].
            foreground_prob_map: foreground probability map. Shape must be [1, H, W].
        """
        if mask.shape[0] != 1 or mask.ndim != 3:
            raise ValueError(f"Input mask should be with size of [1, H, W], but got {mask.shape}")
        if foreground_prob_map.shape[0] != 1 or foreground_prob_map.ndim != 3:
            raise ValueError(
                f"Input foreground_prob_map should be with size of [1, H, W], but got {foreground_prob_map.shape}"
            )

        distance_map = (1.0 - foreground_prob_map) * mask

        if callable(self.smooth_fn):
            distance_map = self.smooth_fn(distance_map)

        return convert_to_dst_type(-distance_map, mask, dtype=self.dtype)[0]


class GenerateMarkers(Transform):
    """
    Generate markers used in `watershed`. Generally, The maximum of this distance (i.e., the minimum of the opposite of
    the distance) are chosen as markers and the flooding of basins from such markers separates the two instances along
    a watershed line. Here is the implementation from HoVerNet papar.
    For more details refer to papers: https://arxiv.org/abs/1812.06499.

    Args:
        threshold: threshold the float values of foreground probability map to int 0 or 1 with specified theashold.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        postprocess_fn: execute additional post transformation on marker. Defaults to None.
        dtype: target data content type to convert, default is np.uint8.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        threshold: float = 0.4,
        radius: int = 2,
        min_size: int = 10,
        remove_small_objects: bool = True,
        postprocess_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.uint8
    ) -> None:
        self.threshold = threshold
        self.radius = radius
        self.postprocess_fn = postprocess_fn
        self.dtype = dtype

        if remove_small_objects:
            self.remove_small_objects = RemoveSmallObjects(min_size=min_size)

    def __call__(self, mask: NdarrayOrTensor, foreground_prob_map: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binarized segmentation result. Shape must be [1, H, W].
            foreground_prob_map: foreground probability map. Shape must be [1, H, W].
        """
        if mask.shape[0] != 1 or mask.ndim != 3:
            raise ValueError(f"Input mask should be with size of [1, H, W], but got {mask.shape}")
        if foreground_prob_map.shape[0] != 1 or foreground_prob_map.ndim != 3:
            raise ValueError(
                f"Input foreground_prob_map should be with size of [1, H, W], but got {foreground_prob_map.shape}"
            )

        foreground_prob_map = foreground_prob_map >= self.threshold  # uncertain area

        marker = mask - convert_to_dst_type(foreground_prob_map, mask, np.uint8)[0]  # certain foreground
        marker[marker < 0] = 0
        if self.postprocess_fn:
            marker = self.postprocess_fn(marker)

        if isinstance(marker, torch.Tensor):
            marker = marker.detach().cpu()

        marker = opening(marker.squeeze(), disk(self.radius))
        marker = label(marker)[0]
        if self.remove_small_objects:
            marker = self.remove_small_objects(marker[None])

        return convert_to_dst_type(marker, mask, dtype=self.dtype)[0]
