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

from typing import Callable, Dict, Hashable, Mapping, Sequence, Union, Optional

import torch

from monai.apps.pathology.transforms.post.array import (
    CalculateInstanceSegmentationMap,
    GenerateMask,
    GenerateProbabilityMap,
    GenerateDistanceMap,
    GenerateMarkers,
)
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from monai.transforms.transform import MapTransform

__all__ = [
    "CalculateInstanceSegmentationMapD",
    "CalculateInstanceSegmentationMapDict",
    "CalculateInstanceSegmentationMapd",
    "GenerateMaskD",
    "GenerateMaskDict",
    "GenerateMaskd",
    "GenerateProbabilityMapD",
    "GenerateProbabilityMapDict",
    "GenerateProbabilityMapd",
    "GenerateDistanceMapD",
    "GenerateDistanceMapDict",
    "GenerateDistanceMapd",
    "GenerateMarkersD",
    "GenerateMarkersDict",
    "GenerateMarkersd",
]


class CalculateInstanceSegmentationMapd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.CalculateInstanceSegmentationMap`.
    Use `skimage.segmentation.watershed` to get instance segmentation results from images.
    See: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        mask_key: keys of mask used in watershed. Only points at which mask == True will be labeled.
        markers_key: keys of markers used in watershed. If None (no markers given), the local minima of the image are used as markers.
        connectivity: An array with the same number of dimensions as image whose non-zero elements indicate neighbors for connection.
            Following the scipy convention, default is a one-connected array of the dimension of the image.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
            ValueError: when the `image` shape is not [1, H, W].
            ValueError: when the `mask` shape is not [1, H, W].

    """

    backend = CalculateInstanceSegmentationMap.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str = "mask",
        markers_key: Optional[str] = None,
        connectivity: Optional[int] = 1,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.markers_key = markers_key
        self.transform = CalculateInstanceSegmentationMap(connectivity=connectivity)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if self.markers_key:
            markers = d[self.markers_key]
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], d[self.mask_key], markers)

        return d


class GenerateMaskd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
        mask_key_postfix: the mask will be written to the value of `{key}_{mask_key_postfix}`.
        softmax: if True, apply a softmax function to the prediction.
        sigmoid: if True, apply a sigmoid function to the prediction.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
            ValueError: when the `image` shape is not [1, H, W].
            ValueError: when the `mask` shape is not [1, H, W].

    """

    backend = GenerateMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key_postfix: str = "mask",
        softmax: bool = False,
        sigmoid: bool = False,
        threshold: Optional[float] = None,
        remove_small_objects: bool = True,
        min_size: int = 10,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key_postfix = mask_key_postfix
        self.transform = GenerateMask(
            softmax = softmax,
            sigmoid = sigmoid,
            threshold = threshold,
            remove_small_objects = remove_small_objects,
            min_size = min_size
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            mask = self.transform(d[key])
            key_to_add = f"{key}_{self.mask_key_postfix}"
            if key_to_add in d:
                raise KeyError(f"Mask with key {key_to_add} already exists.")
            d[key_to_add] = mask
        return d


class GenerateProbabilityMapd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateProbabilityMap`.

    Args:
        keys: keys of the corresponding items to be transformed.
        hover_map_key: keys of hover map used to generate probability map.
        prob_key_postfix: the foreground probability map will be written to the value of `{key}_{prob_key_postfix}`.
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        remove_small_objects: whether need to remove some objects in segmentation results. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateProbabilityMap.backend

    def __init__(
        self,
        keys: KeysCollection,
        hover_map_key: str = "hover_map",
        prob_key_postfix: str = "prob",
        kernel_size: int = 21,
        min_size: int = 10,
        remove_small_objects: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.hover_map_key = hover_map_key
        self.prob_key_postfix = prob_key_postfix
        self.transform = GenerateProbabilityMap(
            kernel_size = kernel_size,
            remove_small_objects = remove_small_objects,
            min_size = min_size,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            prob_map = self.transform(d[key], d[self.hover_map_key])
            key_to_add = f"{key}_{self.prob_key_postfix}"
            if key_to_add in d:
                raise KeyError(f"Probability map with key {key_to_add} already exists.")
            d[key_to_add] = prob_map
        return d


class GenerateDistanceMapd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateDistanceMap`.

    Args:
        keys: keys of the corresponding items to be transformed.
        prob_key: keys of the foreground probability map used to generate distance map.
        dist_key_postfix: the distance map will be written to the value of `{key}_{dist_key_postfix}`.
        sigma: std. Used in `GaussianFilter`. Could be a single value, or `spatial_dims` number of values. Defaults to 0.4.
        smooth_fn: execute smooth function on distance map. Defaults to None. You can specify "guassian" or other callable functions.
            If specify "guassian", `GaussianFilter` will applied on distance map.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateDistanceMap.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob_key: str = "prob",
        dist_key_postfix: str = "dist",
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.4,
        smooth_fn: Optional[Union[Callable, str]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob_key = prob_key
        self.dist_key_postfix = dist_key_postfix
        self.transform = GenerateDistanceMap(
            sigma = sigma,
            smooth_fn = smooth_fn
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            distance_map = self.transform(d[key], d[self.prob_key])
            key_to_add = f"{key}_{self.dist_key_postfix}"
            if key_to_add in d:
                raise KeyError(f"Distance map with key {key_to_add} already exists.")
            d[key_to_add] = distance_map
        return d


class GenerateMarkersd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateMarkers`.

    Args:
        keys: keys of the corresponding items to be transformed.
        prob_key: keys of the foreground probability map used to generate markers.
        markers_key_postfix: the markers will be written to the value of `{key}_{markers_key_postfix}`.
        threshold: threshold the float values of foreground probability map to int 0 or 1 with specified theashold.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        postprocess_fn: execute additional post transformation on marker. Defaults to None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateMarkers.backend

    def __init__(
        self,
        keys: KeysCollection,
        prob_key: str = "prob",
        markers_key_postfix: str = "markers",
        threshold: float = 0.4,
        radius: int = 2,
        min_size: int = 10,
        remove_small_objects: bool = True,
        postprocess_fn: Callable = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.prob_key = prob_key
        self.markers_key_postfix = markers_key_postfix
        self.transform = GenerateMarkers(
            threshold = threshold,
            radius = radius,
            min_size = min_size,
            remove_small_objects = remove_small_objects,
            postprocess_fn = postprocess_fn,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            markers = self.transform(d[key], d[self.prob_key])
            key_to_add = f"{key}_{self.markers_key_postfix}"
            if key_to_add in d:
                raise KeyError(f"Markers with key {key_to_add} already exists.")
            d[key_to_add] = markers
        return d


CalculateInstanceSegmentationMapD = CalculateInstanceSegmentationMapDict = CalculateInstanceSegmentationMapd
GenerateMaskD = GenerateMaskDict = GenerateMaskd
GenerateProbabilityMapD = GenerateProbabilityMapDict = GenerateProbabilityMapd
GenerateDistanceMapD = GenerateDistanceMapDict = GenerateDistanceMapd
GenerateMarkersD = GenerateMarkersDict = GenerateMarkersd
