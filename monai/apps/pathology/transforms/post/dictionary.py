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

from typing import Callable, Dict, Hashable, Mapping, Optional

import numpy as np

from monai.apps.pathology.transforms.post.array import (
    GenerateDistanceMap,
    GenerateInstanceBorder,
    GenerateWatershedMarkers,
    GenerateWatershedMask,
    Watershed,
    GenerateInstanceCentroid,
    GenerateInstanceContour,
    GenerateInstanceType,
    GenerateSuccinctContour,
)
from monai.config.type_definitions import DtypeLike, KeysCollection, NdarrayOrTensor
from monai.transforms.transform import MapTransform
from monai.utils import optional_import

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
        softmax: if True, apply a softmax function to the prediction.
        sigmoid: if True, apply a sigmoid function to the prediction.
        threshold: if not None, threshold the float values to int number 0 or 1 with specified theashold.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        dtype: target data content type to convert. Defaults to np.uint8.
        allow_missing_keys: don't raise exception if key is missing.

    """

    backend = GenerateWatershedMask.backend

    def __init__(
        self,
        keys: KeysCollection,
        mask_key: str = "mask",
        softmax: bool = True,
        sigmoid: bool = False,
        threshold: Optional[float] = None,
        remove_small_objects: bool = True,
        min_size: int = 10,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mask_key = mask_key
        self.transform = GenerateWatershedMask(
            softmax=softmax,
            sigmoid=sigmoid,
            threshold=threshold,
            remove_small_objects=remove_small_objects,
            min_size=min_size,
            dtype=dtype,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            mask = self.transform(d[key])
            key_to_add = f"{self.mask_key}"
            if key_to_add in d:
                raise KeyError(f"Mask with key {key_to_add} already exists.")
            d[key_to_add] = mask
        return d


class GenerateInstanceBorderd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateInstanceBorder`.

    Args:
        keys: keys of the corresponding items to be transformed.
        hover_map_key: keys of hover map used to generate probability map.
        border_key: the instance border map will be written to the value of `{border_key}`.
        kernel_size: the size of the Sobel kernel. Defaults to 21.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in segmentation results. Defaults to True.
        dtype: target data content type to convert, default is np.float32.
        allow_missing_keys: don't raise exception if key is missing.

    Raises:
        ValueError: when the `hover_map` has only one value.
        ValueError: when the `sobel gradient map` has only one value.

    """

    backend = GenerateInstanceBorder.backend

    def __init__(
        self,
        keys: KeysCollection,
        hover_map_key: str = "hover_map",
        border_key: str = "border",
        kernel_size: int = 21,
        min_size: int = 10,
        remove_small_objects: bool = True,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.hover_map_key = hover_map_key
        self.border_key = border_key
        self.transform = GenerateInstanceBorder(
            kernel_size=kernel_size, remove_small_objects=remove_small_objects, min_size=min_size, dtype=dtype
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            instance_border = self.transform(d[key], d[self.hover_map_key])
            key_to_add = f"{self.border_key}"
            if key_to_add in d:
                raise KeyError(f"Instance border map with key {key_to_add} already exists.")
            d[key_to_add] = instance_border
        return d


class GenerateDistanceMapd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateDistanceMap`.

    Args:
        keys: keys of the corresponding items to be transformed.
        border_key: keys of the instance border map used to generate distance map.
        dist_key: the distance map will be written to the value of `{dist_key}`.
        smooth_fn: execute smooth function on distance map. Defaults to None. You can specify
            callable functions for smoothing.
            For example, if you want apply gaussian smooth, you can specify `smooth_fn = GaussianSmooth()`
        dtype: target data content type to convert, default is np.float32.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateDistanceMap.backend

    def __init__(
        self,
        keys: KeysCollection,
        border_key: str = "border",
        dist_key: str = "dist",
        smooth_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.border_key = border_key
        self.dist_key = dist_key
        self.transform = GenerateDistanceMap(smooth_fn=smooth_fn, dtype=dtype)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            distance_map = self.transform(d[key], d[self.border_key])
            key_to_add = f"{self.dist_key}"
            if key_to_add in d:
                raise KeyError(f"Distance map with key {key_to_add} already exists.")
            d[key_to_add] = distance_map
        return d


class GenerateWatershedMarkersd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.pathology.transforms.array.GenerateWatershedMarkers`.

    Args:
        keys: keys of the corresponding items to be transformed.
        border_key: keys of the instance border map used to generate markers.
        markers_key: the markers will be written to the value of `{markers_key}`.
        threshold: threshold the float values of instance border map to int 0 or 1 with specified theashold.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_size: objects smaller than this size are removed if `remove_small_objects` is True. Defaults to 10.
        remove_small_objects: whether need to remove some objects in the marker. Defaults to True.
        postprocess_fn: execute additional post transformation on marker. Defaults to None.
        dtype: target data content type to convert, default is np.uint8.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = GenerateWatershedMarkers.backend

    def __init__(
        self,
        keys: KeysCollection,
        border_key: str = "border",
        markers_key: str = "markers",
        threshold: float = 0.4,
        radius: int = 2,
        min_size: int = 10,
        remove_small_objects: bool = True,
        postprocess_fn: Optional[Callable] = None,
        dtype: DtypeLike = np.uint8,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.border_key = border_key
        self.markers_key = markers_key
        self.transform = GenerateWatershedMarkers(
            threshold=threshold,
            radius=radius,
            min_size=min_size,
            remove_small_objects=remove_small_objects,
            postprocess_fn=postprocess_fn,
            dtype=dtype,
        )

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            markers = self.transform(d[key], d[self.border_key])
            key_to_add = f"{self.markers_key}"
            if key_to_add in d:
                raise KeyError(f"Markers with key {key_to_add} already exists.")
            d[key_to_add] = markers
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
        points_num: assumed that the created contour does not form a contour if it does not contain more points
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
        points_num: int = 3,
        level: Optional[float] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.converter = GenerateInstanceContour(points_num=points_num, level=level)
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



WatershedD = WatershedDict = Watershedd
GenerateWatershedMaskD = GenerateWatershedMaskDict = GenerateWatershedMaskd
GenerateInstanceBorderD = GenerateInstanceBorderDict = GenerateInstanceBorderd
GenerateDistanceMapD = GenerateDistanceMapDict = GenerateDistanceMapd
GenerateWatershedMarkersD = GenerateWatershedMarkersDict = GenerateWatershedMarkersd
GenerateSuccinctContourDict = GenerateSuccinctContourD = GenerateSuccinctContourd
GenerateInstanceContourDict = GenerateInstanceContourD = GenerateInstanceContourd
GenerateInstanceCentroidDict = GenerateInstanceCentroidD = GenerateInstanceCentroidd
GenerateInstanceTypeDict = GenerateInstanceTypeD = GenerateInstanceTyped
