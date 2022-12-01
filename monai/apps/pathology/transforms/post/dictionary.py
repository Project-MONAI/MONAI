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

from typing import Callable, Dict, Hashable, Mapping, Optional, Union

import numpy as np

from monai.apps.pathology.transforms.post.array import (
    GenerateDistanceMap,
    GenerateInstanceBorder,
    GenerateInstanceCentroid,
    GenerateInstanceContour,
    GenerateInstanceType,
    GenerateSuccinctContour,
    GenerateWatershedMarkers,
    GenerateWatershedMask,
    HoVerNetTypeMapPostProcessing,
    Watershed,
)
from monai.config.type_definitions import DtypeLike, KeysCollection, NdarrayOrTensor
from monai.transforms.transform import MapTransform, Transform
from monai.utils import optional_import
from monai.utils.enums import HoVerNetBranch

find_contours, _ = optional_import("skimage.measure", name="find_contours")
moments, _ = optional_import("skimage.measure", name="moments")

__all__ = [
    "WatershedD",
    "WatershedDict",
    "Watershedd",
    "GenerateWatershedMaskD",
    "GenerateWatershedMaskDict",
    "GenerateWatershedMaskd",
    "GenerateInstanceBorderD",
    "GenerateInstanceBorderDict",
    "GenerateInstanceBorderd",
    "GenerateDistanceMapD",
    "GenerateDistanceMapDict",
    "GenerateDistanceMapd",
    "GenerateWatershedMarkersD",
    "GenerateWatershedMarkersDict",
    "GenerateWatershedMarkersd",
    "GenerateSuccinctContourDict",
    "GenerateSuccinctContourD",
    "GenerateSuccinctContourd",
    "GenerateInstanceContourDict",
    "GenerateInstanceContourD",
    "GenerateInstanceContourd",
    "GenerateInstanceCentroidDict",
    "GenerateInstanceCentroidD",
    "GenerateInstanceCentroidd",
    "GenerateInstanceTypeDict",
    "GenerateInstanceTypeD",
    "GenerateInstanceTyped",
    "HoVerNetTypeMapPostProcessingDict",
    "HoVerNetTypeMapPostProcessingD",
    "HoVerNetTypeMapPostProcessingd",
]


class Watershedd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.Watershed`.
    Use `skimage.segmentation.watershed` to get instance segmentation results from images.
    See: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: monai.transforms.MapTransform
        mask_key: keys of mask used in watershed. Only points at which mask == True will be labeled.
        markers_key: keys of markers used in watershed. If None (no markers given), the local minima of the image are
            used as markers.
        connectivity: An array with the same number of dimensions as image whose non-zero elements indicate neighbors
            for connection. Following the scipy convention, default is a one-connected array of the dimension of the
            image.
        dtype: target data content type to convert. Defaults to np.uint8.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: when the `image` shape is not [1, H, W].
        ValueError: when the `mask` shape is not [1, H, W].

    """

    backend = Watershed.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: Optional[str] = "mask",
        markers_key: Optional[str] = None,
        connectivity: Optional[int] = 1,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.markers_key = markers_key
        self.transform = Watershed(connectivity=connectivity, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        markers = d[self.markers_key] if self.markers_key else None
        mask = d[self.mask_key] if self.mask_key else None

        for key in self.key_iterator(d):
            d[key] = self.transform(d[key], mask, markers)

        return d


class GenerateWatershedMaskd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateWatershedMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
        mask_key: the mask will be written to the value of `{mask_key}`.
        activation: the activation layer to be applied on nuclear type branch. It can be "softmax" or "sigmoid" string,
            or any callable. Defaults to "softmax".
        threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold.
        min_object_size: objects smaller than this size are removed. Defaults to 10.
        dtype: target data content type to convert, default is np.uint8.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = GenerateWatershedMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str = "mask",
        activation: Union[str, Callable] = "softmax",
        threshold: Optional[float] = None,
        min_object_size: int = 10,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.transform = GenerateWatershedMask(
            activation=activation,
            threshold=threshold,
            min_object_size=min_object_size,
            dtype=dtype,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            mask = self.transform(d[key])
            if self.mask_key in d:
                raise KeyError(f"Mask with key {self.mask_key} already exists.")
            d[self.mask_key] = mask
        return d


class GenerateInstanceBorderd(Transform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateInstanceBorder`.

    Args:
        mask_key: the input key where the watershed mask is stored. Defaults to `"mask"`.
        hover_map_key: the input key where hover map is stored. Defaults to `"hover_map"`.
        border_key: the output key where instance border map is written. Defaults to `"border"`.
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        dtype: target data content type to convert, default is np.float32.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: when the `hover_map` has only one value.
        ValueError: when the `sobel gradient map` has only one value.

    """

    backend = GenerateInstanceBorder.backend

    def __init__(
        self,
        mask_key: str = "mask",
        hover_map_key: str = "hover_map",
        border_key: str = "border",
        kernel_size: int = 21,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.mask_key = mask_key
        self.hover_map_key = hover_map_key
        self.border_key = border_key
        self.transform = GenerateInstanceBorder(kernel_size=kernel_size, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if self.border_key in d:
            raise KeyError(f"The key '{self.border_key}' for instance border map already exists.")
        d[self.border_key] = self.transform(d[self.mask_key], d[self.hover_map_key])
        return d


class GenerateDistanceMapd(Transform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateDistanceMap`.

    Args:
        mask_key: the input key where the watershed mask is stored. Defaults to `"mask"`.
        border_key: the input key where instance border map is stored. Defaults to `"border"`.
        dist_map_key: the output key where distance map is written. Defaults to `"dist_map"`.
        smooth_fn: smoothing function for distance map, which can be any callable object.
            If not provided :py:class:`monai.transforms.GaussianSmooth()` is used.
        dtype: target data content type to convert, default is np.float32.
    """

    backend = GenerateDistanceMap.backend

    def __init__(
        self,
        mask_key: str = "mask",
        border_key: str = "border",
        dist_map_key: str = "dist_map",
        smooth_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.float32,
    ) -> None:
        self.mask_key = mask_key
        self.border_key = border_key
        self.dist_map_key = dist_map_key
        self.transform = GenerateDistanceMap(smooth_fn=smooth_fn, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if self.dist_map_key in d:
            raise KeyError(f"The key '{self.dist_map_key}' for distance map already exists.")
        d[self.dist_map_key] = self.transform(d[self.mask_key], d[self.border_key])
        return d


class GenerateWatershedMarkersd(Transform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateWatershedMarkers`.

    Args:
        mask_key: the input key where the watershed mask is stored. Defaults to `"mask"`.
        border_key: the input key where instance border map is stored. Defaults to `"border"`.
        markers_key: the output key where markers is written. Defaults to `"markers"`.
        threshold: threshold the float values of instance border map to int 0 or 1 with specified threshold.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_object_size: objects smaller than this size are removed. Defaults to 10.
        postprocess_fn: execute additional post transformation on marker. Defaults to None.
        dtype: target data content type to convert, default is np.uint8.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateWatershedMarkers.backend

    def __init__(
        self,
        mask_key: str = "mask",
        border_key: str = "border",
        markers_key: str = "markers",
        threshold: float = 0.4,
        radius: int = 2,
        min_object_size: int = 10,
        postprocess_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.uint8,
    ) -> None:
        self.mask_key = mask_key
        self.border_key = border_key
        self.markers_key = markers_key
        self.transform = GenerateWatershedMarkers(
            threshold=threshold,
            radius=radius,
            min_object_size=min_object_size,
            postprocess_fn=postprocess_fn,
            dtype=dtype,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        if self.markers_key in d:
            raise KeyError(f"The key '{self.markers_key}' for markers already exists.")
        d[self.markers_key] = self.transform(d[self.mask_key], d[self.border_key])
        return d


class GenerateSuccinctContourd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.post.array.GenerateSuccinctContour`.
    Converts Scipy-style contours(generated by skimage.measure.find_contours) to a more succinct version which
    only includes the pixels to which lines need to be drawn (i.e. not the intervening pixels along each line).

    Args:
        keys: keys of the corresponding items to be transformed.
        height: height of bounding box, used to detect direction of line segment.
        width: width of bounding box, used to detect direction of line segment.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = GenerateSuccinctContour.backend

    def __init__(self, keys: KeysCollection, height: int, width: int, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GenerateSuccinctContour(height=height, width=width)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])

        return d


class GenerateInstanceContourd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.post.array.GenerateInstanceContour`.
    Generate contour for each instance in a 2D array. Use `GenerateSuccinctContour` to only include the pixels
    to which lines need to be drawn

    Args:
        keys: keys of the corresponding items to be transformed.
        contour_key_postfix: the output contour coordinates will be written to the value of
            `{key}_{contour_key_postfix}`.
        offset_key: keys of offset used in `GenerateInstanceContour`.
        min_num_points: assumed that the created contour does not form a contour if it does not contain more points
            than the specified value. Defaults to 3.
        level: optional. Value along which to find contours in the array. By default, the level is set
            to (max(image) + min(image)) / 2.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = GenerateInstanceContour.backend

    def __init__(
        self,
        keys: KeysCollection,
        contour_key_postfix: str = "contour",
        offset_key: Optional[str] = None,
        min_num_points: int = 3,
        level: Optional[float] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GenerateInstanceContour(min_num_points=min_num_points, level=level)
        self.contour_key_postfix = contour_key_postfix
        self.offset_key = offset_key

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            offset = d[self.offset_key] if self.offset_key else None
            contour = self.converter(d[key], offset)
            key_to_add = f"{key}_{self.contour_key_postfix}"
            if key_to_add in d:
                raise KeyError(f"Contour with key {key_to_add} already exists.")
            d[key_to_add] = contour
        return d


class GenerateInstanceCentroidd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.post.array.GenerateInstanceCentroid`.
    Generate instance centroid using `skimage.measure.centroid`.

    Args:
        keys: keys of the corresponding items to be transformed.
        centroid_key_postfix: the output centroid coordinates will be written to the value of
            `{key}_{centroid_key_postfix}`.
        offset_key: keys of offset used in `GenerateInstanceCentroid`.
        dtype: the data type of output centroid.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = GenerateInstanceCentroid.backend

    def __init__(
        self,
        keys: KeysCollection,
        centroid_key_postfix: str = "centroid",
        offset_key: Optional[str] = None,
        dtype: Optional[DtypeLike] = int,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GenerateInstanceCentroid(dtype=dtype)
        self.centroid_key_postfix = centroid_key_postfix
        self.offset_key = offset_key

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            offset = d[self.offset_key] if self.offset_key else None
            centroid = self.converter(d[key], offset)
            key_to_add = f"{key}_{self.centroid_key_postfix}"
            if key_to_add in d:
                raise KeyError(f"Centroid with key {key_to_add} already exists.")
            d[key_to_add] = centroid
        return d


class GenerateInstanceTyped(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.post.array.GenerateInstanceType`.
    Generate instance type and probability for each instance.

    Args:
        keys: keys of the corresponding items to be transformed.
        type_info_key: the output instance type and probability will be written to the value of
            `{type_info_key}`.
        bbox_key: keys of bounding box.
        seg_pred_key: keys of segmentation prediction map.
        instance_id_key: keys of instance id.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = GenerateInstanceType.backend

    def __init__(
        self,
        keys: KeysCollection,
        type_info_key: str = "type_info",
        bbox_key: str = "bbox",
        seg_pred_key: str = "seg",
        instance_id_key: str = "id",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GenerateInstanceType()
        self.type_info_key = type_info_key
        self.bbox_key = bbox_key
        self.seg_pred_key = seg_pred_key
        self.instance_id_key = instance_id_key

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            seg = d[self.seg_pred_key]
            bbox = d[self.bbox_key]
            id = d[self.instance_id_key]
            instance_type, type_prob = self.converter(d[key], seg, bbox, id)
            key_to_add = f"{self.type_info_key}"
            if key_to_add in d:
                raise KeyError(f"Type information with key {key_to_add} already exists.")
            d[key_to_add] = {"inst_type": instance_type, "type_prob": type_prob}
        return d


class HoVerNetTypeMapPostProcessingd(Transform):
    """
    Dictionary-based wrapper for :py:class:`monai.apps.pathology.transforms.post.array.HoVerNetTypeMapPostProcessing`.
    It generate a dictionary containing centroid, bounding box, type prediction for each instance. Also if requested,
    it returns binary maps for instance segmentation and predicted types.

    The output dictionary will have three new keys:

    - "instance_info" mapping each instance to their centroid, bounding box, predicted type and probability.
    - "instance_seg_map" containing an instance segmentation map.
    - "type_seg_map" containing a segmentation map with associated types for each pixel.

    Args:
        level: optional value for `skimage.measure.find_contours`, to find contours in the array.
            If not provided, the level is set to (max(image) + min(image)) / 2.
        distance_smooth_fn: smoothing function for distance map.
            If not provided, :py:class:`monai.transforms.intensity.GaussianSmooth()` will be used..
        marker_postprocess_fn: post-process function for watershed markers.
            If not provided, :py:class:`monai.transforms.post.FillHoles()` will be used.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(
        self,
        nuclear_prediction_key: str = HoVerNetBranch.NP.value,
        hover_map_key: str = HoVerNetBranch.HV.value,
        type_prediction_key: str = HoVerNetBranch.NC.value,
        min_num_points: int = 3,
        level: Optional[float] = None,
        distance_smooth_fn: Optional[Callable] = None,
        marker_postprocess_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.post_process = HoVerNetTypeMapPostProcessing(
            min_num_points=min_num_points,
            level=level,
            distance_smooth_fn=distance_smooth_fn,
            marker_postprocess_fn=marker_postprocess_fn,
        )
        self.nuclear_prediction_key = nuclear_prediction_key
        self.hover_map_key = hover_map_key
        self.type_prediction_key = type_prediction_key

    def __call__(self, data):
        d = dict(data)
        instance_info, instance_seg_map, type_seg_map = self.post_process(
            d[self.nuclear_prediction_key], d[self.hover_map_key], d[self.type_prediction_key]
        )

        output_keys = ["instance_info", "instance_seg_map", "type_seg_map"]
        for k in output_keys:
            if k in d:
                raise ValueError("The output key ['{k}'] already exists in the input dictionary!")

        d["instance_info"] = instance_info
        d["instance_seg_map"] = instance_seg_map
        d["type_seg_map"] = type_seg_map

        return d


WatershedD = WatershedDict = Watershedd
GenerateWatershedMaskD = GenerateWatershedMaskDict = GenerateWatershedMaskd
GenerateInstanceBorderD = GenerateInstanceBorderDict = GenerateInstanceBorderd
GenerateDistanceMapD = GenerateDistanceMapDict = GenerateDistanceMapd
GenerateWatershedMarkersD = GenerateWatershedMarkersDict = GenerateWatershedMarkersd
GenerateSuccinctContourDict = GenerateSuccinctContourD = GenerateSuccinctContourd
GenerateInstanceContourDict = GenerateInstanceContourD = GenerateInstanceContourd
GenerateInstanceCentroidDict = GenerateInstanceCentroidD = GenerateInstanceCentroidd
GenerateInstanceTypeDict = GenerateInstanceTypeD = GenerateInstanceTyped
HoVerNetTypeMapPostProcessingDict = HoVerNetTypeMapPostProcessingD = HoVerNetTypeMapPostProcessingd
