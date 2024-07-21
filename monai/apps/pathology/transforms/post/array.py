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

from __future__ import annotations

import warnings
from typing import Callable, Sequence

import numpy as np
import torch

from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.transforms import (
    Activations,
    AsDiscrete,
    BoundingRect,
    FillHoles,
    GaussianSmooth,
    RemoveSmallObjects,
    SobelGradients,
)
from monai.transforms.transform import Transform
from monai.transforms.utils_pytorch_numpy_unification import max, maximum, min, sum, unique
from monai.utils import TransformBackends, convert_to_numpy, optional_import
from monai.utils.misc import ensure_tuple_rep
from monai.utils.type_conversion import convert_to_dst_type, convert_to_tensor

label, _ = optional_import("scipy.ndimage", name="label")
disk, _ = optional_import("skimage.morphology", name="disk")
opening, _ = optional_import("skimage.morphology", name="opening")
watershed, _ = optional_import("skimage.segmentation", name="watershed")
find_contours, _ = optional_import("skimage.measure", name="find_contours")
centroid, _ = optional_import("skimage.measure", name="centroid")

__all__ = [
    "Watershed",
    "GenerateWatershedMask",
    "GenerateInstanceBorder",
    "GenerateDistanceMap",
    "GenerateWatershedMarkers",
    "GenerateSuccinctContour",
    "GenerateInstanceContour",
    "GenerateInstanceCentroid",
    "GenerateInstanceType",
    "HoVerNetInstanceMapPostProcessing",
    "HoVerNetNuclearTypePostProcessing",
]


class Watershed(Transform):
    """
    Use `skimage.segmentation.watershed` to get instance segmentation results from images.
    See: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed.

    Args:
        connectivity: an array with the same number of dimensions as image whose non-zero elements indicate
            neighbors for connection. Following the scipy convention, default is a one-connected array of
            the dimension of the image.
        dtype: target data content type to convert, default is np.int64.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, connectivity: int | None = 1, dtype: DtypeLike = np.int64) -> None:
        self.connectivity = connectivity
        self.dtype = dtype

    def __call__(
        self, image: NdarrayOrTensor, mask: NdarrayOrTensor | None = None, markers: NdarrayOrTensor | None = None
    ) -> NdarrayOrTensor:
        """
        Args:
            image: image where the lowest value points are labeled first. Shape must be [1, H, W, [D]].
            mask: optional, the same shape as image. Only points at which mask == True will be labeled.
                If None (no mask given), it is a volume of all 1s.
            markers: optional, the same shape as image. The desired number of markers, or an array marking
                the basins with the values to be assigned in the label matrix. Zero means not a marker.
                If None (no markers given), the local minima of the image are used as markers.
        """

        image = convert_to_numpy(image)
        markers = convert_to_numpy(markers)
        mask = convert_to_numpy(mask)

        instance_seg = watershed(image, markers=markers, mask=mask, connectivity=self.connectivity)

        return convert_to_dst_type(instance_seg, image, dtype=self.dtype)[0]


class GenerateWatershedMask(Transform):
    """
    generate mask used in `watershed`. Only points at which mask == True will be labeled.

    Args:
        activation: the activation layer to be applied on the input probability map.
            It can be "softmax" or "sigmoid" string, or any callable. Defaults to "softmax".
        threshold: an optional float value to threshold to binarize probability map.
            If not provided, defaults to 0.5 when activation is not "softmax", otherwise None.
        min_object_size: objects smaller than this size (in pixel) are removed. Defaults to 10.
        dtype: target data content type to convert, default is np.uint8.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        activation: str | Callable = "softmax",
        threshold: float | None = None,
        min_object_size: int = 10,
        dtype: DtypeLike = np.uint8,
    ) -> None:
        self.dtype = dtype

        # set activation layer
        use_softmax = False
        use_sigmoid = False
        activation_fn = None
        if isinstance(activation, str):
            if activation.lower() == "softmax":
                use_softmax = True
            elif activation.lower() == "sigmoid":
                use_sigmoid = True
            else:
                raise ValueError(
                    f"The activation should be 'softmax' or 'sigmoid' string, or any callable. '{activation}' was given."
                )
        elif callable(activation):
            activation_fn = activation
        else:
            raise ValueError(f"The activation type should be either str or callable. '{type(activation)}' was given.")
        self.activation = Activations(softmax=use_softmax, sigmoid=use_sigmoid, other=activation_fn)

        # set discretization transform
        if not use_softmax and threshold is None:
            threshold = 0.5
        self.as_discrete = AsDiscrete(threshold=threshold, argmax=use_softmax)

        # set small object removal transform
        self.remove_small_objects = RemoveSmallObjects(min_size=min_object_size) if min_object_size > 0 else None

    def __call__(self, prob_map: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Args:
            prob_map: probability map of segmentation, shape must be [C, H, W, [D]]
        """

        pred = self.activation(prob_map)
        pred = self.as_discrete(pred)

        pred = convert_to_numpy(pred)

        pred = label(pred)[0]
        if self.remove_small_objects is not None:
            pred = self.remove_small_objects(pred)
        pred[pred > 0] = 1

        return convert_to_dst_type(pred, prob_map, dtype=self.dtype)[0]


class GenerateInstanceBorder(Transform):
    """
    Generate instance border by hover map. The more parts of the image that cannot be identified as foreground areas,
    the larger the grey scale value. The grey value of the instance's border will be larger.

    Args:
        kernel_size: the size of the Sobel kernel. Defaults to 5.
        dtype: target data type to convert to. Defaults to np.float32.


    Raises:
        ValueError: when the `mask` shape is not [1, H, W].
        ValueError: when the `hover_map` shape is not [2, H, W].

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, kernel_size: int = 5, dtype: DtypeLike = np.float32) -> None:
        self.dtype = dtype
        self.sobel_gradient = SobelGradients(kernel_size=kernel_size)

    def __call__(self, mask: NdarrayOrTensor, hover_map: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binary segmentation map, the output of :py:class:`GenerateWatershedMask`.
                Shape must be [1, H, W] or [H, W].
            hover_map:  horizontal and vertical distances of nuclear pixels to their centres of mass. Shape must be [2, H, W].
                The first and second channel represent the horizontal and vertical maps respectively. For more details refer
                to papers: https://arxiv.org/abs/1812.06499.
        """
        if len(hover_map.shape) != 3:
            raise ValueError(f"The hover map should have the shape of [C, H, W], but got {hover_map.shape}.")
        if len(mask.shape) == 3:
            if mask.shape[0] != 1:
                raise ValueError(f"The mask should have only one channel, but got {mask.shape[0]}.")
        elif len(mask.shape) == 2:
            mask = mask[None]
        else:
            raise ValueError(f"The mask should have the shape of [1, H, W] or [H, W], but got {mask.shape}.")
        if hover_map.shape[0] != 2:
            raise ValueError(f"Suppose the hover map only has two channels, but got {hover_map.shape[0]}")

        hover_h = hover_map[0:1, ...]
        hover_v = hover_map[1:2, ...]

        hover_h_min, hover_h_max = min(hover_h), max(hover_h)
        hover_v_min, hover_v_max = min(hover_v), max(hover_v)
        if (hover_h_max - hover_h_min) == 0 or (hover_v_max - hover_v_min) == 0:
            raise ValueError("Not a valid hover map, please check your input")
        hover_h = (hover_h - hover_h_min) / (hover_h_max - hover_h_min)
        hover_v = (hover_v - hover_v_min) / (hover_v_max - hover_v_min)
        sobelh = self.sobel_gradient(hover_h)[1, ...]
        sobelv = self.sobel_gradient(hover_v)[0, ...]
        sobelh_min, sobelh_max = min(sobelh), max(sobelh)
        sobelv_min, sobelv_max = min(sobelv), max(sobelv)
        if (sobelh_max - sobelh_min) == 0 or (sobelv_max - sobelv_min) == 0:
            raise ValueError("Not a valid sobel gradient map")
        sobelh = 1 - (sobelh - sobelh_min) / (sobelh_max - sobelh_min)
        sobelv = 1 - (sobelv - sobelv_min) / (sobelv_max - sobelv_min)

        # combine the h & v values using max
        overall = maximum(sobelh, sobelv)
        overall = overall - (1 - mask)
        overall[overall < 0] = 0

        return convert_to_dst_type(overall, mask, dtype=self.dtype)[0]


class GenerateDistanceMap(Transform):
    """
    Generate distance map.
    In general, the instance map is calculated from the distance to the background.
    Here, we use 1 - "instance border map" to generate the distance map.
    Nuclei values form mountains so invert them to get basins.

    Args:
        smooth_fn: smoothing function for distance map, which can be any callable object.
            If not provided :py:class:`monai.transforms.GaussianSmooth()` is used.
        dtype: target data type to convert to. Defaults to np.float32.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, smooth_fn: Callable | None = None, dtype: DtypeLike = np.float32) -> None:
        self.smooth_fn = smooth_fn if smooth_fn is not None else GaussianSmooth()
        self.dtype = dtype

    def __call__(self, mask: NdarrayOrTensor, instance_border: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binary segmentation map, the output of :py:class:`GenerateWatershedMask`.
                Shape must be [1, H, W] or [H, W].
            instance_border: instance border map, the output of :py:class:`GenerateInstanceBorder`.
                Shape must be [1, H, W].
        """
        if len(mask.shape) == 3:
            if mask.shape[0] != 1:
                raise ValueError(f"The mask should have only one channel, but got {mask.shape[0]}.")
        elif len(mask.shape) == 2:
            mask = mask[None]
        else:
            raise ValueError(f"The mask should have the shape of [1, H, W] or [H, W], but got {mask.shape}.")
        if instance_border.shape[0] != 1 or instance_border.ndim != 3:
            raise ValueError(f"Input instance_border should be with size of [1, H, W], but got {instance_border.shape}")

        distance_map = (1.0 - instance_border) * mask
        distance_map = self.smooth_fn(distance_map)  # type: ignore

        return convert_to_dst_type(-distance_map, mask, dtype=self.dtype)[0]


class GenerateWatershedMarkers(Transform):
    """
    Generate markers to be used in `watershed`. The watershed algorithm treats pixels values as a local topography
    (elevation). The algorithm floods basins from the markers until basins attributed to different markers meet on
    watershed lines. Generally, markers are chosen as local minima of the image, from which basins are flooded.
    Here is the implementation from HoVerNet paper.
    For more details refer to papers: https://arxiv.org/abs/1812.06499.

    Args:
        threshold: a float value to threshold to binarize instance border map.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        radius: the radius of the disk-shaped footprint used in `opening`. Defaults to 2.
        min_object_size: objects smaller than this size (in pixel) are removed. Defaults to 10.
        postprocess_fn: additional post-process function on the markers.
            If not provided, :py:class:`monai.transforms.post.FillHoles()` will be used.
        dtype: target data type to convert to. Defaults to np.int64.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        threshold: float = 0.4,
        radius: int = 2,
        min_object_size: int = 10,
        postprocess_fn: Callable | None = None,
        dtype: DtypeLike = np.int64,
    ) -> None:
        self.threshold = threshold
        self.radius = radius
        self.dtype = dtype
        if postprocess_fn is None:
            postprocess_fn = FillHoles()

        self.postprocess_fn = postprocess_fn
        self.remove_small_objects = RemoveSmallObjects(min_size=min_object_size) if min_object_size > 0 else None

    def __call__(self, mask: NdarrayOrTensor, instance_border: NdarrayOrTensor) -> NdarrayOrTensor:  # type: ignore
        """
        Args:
            mask: binary segmentation map, the output of :py:class:`GenerateWatershedMask`.
                Shape must be [1, H, W] or [H, W].
            instance_border: instance border map, the output of :py:class:`GenerateInstanceBorder`.
                Shape must be [1, H, W].
        """
        if len(mask.shape) == 3:
            if mask.shape[0] != 1:
                raise ValueError(f"The mask should have only one channel, but got {mask.shape[0]}.")
        elif len(mask.shape) == 2:
            mask = mask[None]
        else:
            raise ValueError(f"The mask should have the shape of [1, H, W] or [H, W], but got {mask.shape}.")
        if instance_border.shape[0] != 1 or instance_border.ndim != 3:
            raise ValueError(f"Input instance_border should be with size of [1, H, W], but got {instance_border.shape}")

        instance_border = instance_border >= self.threshold  # uncertain area

        marker = mask - convert_to_dst_type(instance_border, mask)[0]  # certain foreground
        marker[marker < 0] = 0
        marker = self.postprocess_fn(marker)
        marker = convert_to_numpy(marker)

        marker = opening(marker.squeeze(), disk(self.radius))
        marker = label(marker)[0][None]
        if self.remove_small_objects is not None:
            marker = self.remove_small_objects(marker)

        return convert_to_dst_type(marker, mask, dtype=self.dtype)[0]


class GenerateSuccinctContour(Transform):
    """
    Converts SciPy-style contours (generated by skimage.measure.find_contours) to a more succinct version which only includes
    the pixels to which lines need to be drawn (i.e. not the intervening pixels along each line).

    Args:
        height: height of bounding box, used to detect direction of line segment.
        width: width of bounding box, used to detect direction of line segment.

    Returns:
        the pixels that need to be joined by straight lines to describe the outmost pixels of the foreground similar to
            OpenCV's cv.CHAIN_APPROX_SIMPLE (counterclockwise)
    """

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width

    def _generate_contour_coord(self, current: np.ndarray, previous: np.ndarray) -> tuple[int, int]:
        """
        Generate contour coordinates. Given the previous and current coordinates of border positions,
        returns the int pixel that marks the extremity of the segmented pixels.

        Args:
            current: coordinates of the current border position.
            previous: coordinates of the previous border position.
        """

        p_delta = (current[0] - previous[0], current[1] - previous[1])
        row, col = -1, -1

        if p_delta in ((0.0, 1.0), (0.5, 0.5), (1.0, 0.0)):
            row = int(current[0] + 0.5)
            col = int(current[1])
        elif p_delta in ((0.0, -1.0), (0.5, -0.5)):
            row = int(current[0])
            col = int(current[1])
        elif p_delta in ((-1, 0.0), (-0.5, -0.5)):
            row = int(current[0])
            col = int(current[1] + 0.5)
        elif p_delta == (-0.5, 0.5):
            row = int(current[0] + 0.5)
            col = int(current[1] + 0.5)

        return row, col

    def _calculate_distance_from_top_left(self, sequence: Sequence[tuple[int, int]]) -> int:
        """
        Each sequence of coordinates describes a boundary between foreground and background starting and ending at two sides
        of the bounding box. To order the sequences correctly, we compute the distance from the top-left of the bounding box
        around the perimeter in a clockwise direction.

        Args:
            sequence: list of border points coordinates.

        Returns:
            the distance round the perimeter of the bounding box from the top-left origin
        """
        distance: int
        first_coord = sequence[0]
        if first_coord[0] == 0:
            distance = first_coord[1]
        elif first_coord[1] == self.width - 1:
            distance = self.width + first_coord[0]
        elif first_coord[0] == self.height - 1:
            distance = 2 * self.width + self.height - first_coord[1]
        else:
            distance = 2 * (self.width + self.height) - first_coord[0]

        return distance

    def __call__(self, contours: list[np.ndarray]) -> np.ndarray:
        """
        Args:
            contours: list of (n, 2)-ndarrays, scipy-style clockwise line segments, with lines separating foreground/background.
                Each contour is an ndarray of shape (n, 2), consisting of n (row, column) coordinates along the contour.
        """
        pixels: list[tuple[int, int]] = []
        sequences = []
        corners = [False, False, False, False]

        for group in contours:
            sequence: list[tuple[int, int]] = []
            last_added = None
            prev = None
            corner = -1

            for i, coord in enumerate(group):
                if i == 0:
                    # originating from the top, so must be heading south east
                    if coord[0] == 0.0:
                        corner = 1
                        pixel = (0, int(coord[1] - 0.5))
                        if pixel[1] == self.width - 1:
                            corners[1] = True
                        elif pixel[1] == 0.0:
                            corners[0] = True
                    # originating from the left, so must be heading north east
                    elif coord[1] == 0.0:
                        corner = 0
                        pixel = (int(coord[0] + 0.5), 0)
                    # originating from the bottom, so must be heading north west
                    elif coord[0] == self.height - 1:
                        corner = 3
                        pixel = (int(coord[0]), int(coord[1] + 0.5))
                        if pixel[1] == self.width - 1:
                            corners[2] = True
                    # originating from the right, so must be heading south west
                    elif coord[1] == self.width - 1:
                        corner = 2
                        pixel = (int(coord[0] - 0.5), int(coord[1]))
                    else:
                        warnings.warn(f"Invalid contour coord {coord} is generated, skip this instance.")
                        return None  # type: ignore
                    sequence.append(pixel)
                    last_added = pixel
                elif i == len(group) - 1:
                    # add this point
                    pixel = self._generate_contour_coord(coord, prev)  # type: ignore
                    if pixel != last_added:
                        sequence.append(pixel)
                        last_added = pixel
                elif np.any(coord - prev != group[i + 1] - coord):
                    pixel = self._generate_contour_coord(coord, prev)  # type: ignore
                    if pixel != last_added:
                        sequence.append(pixel)
                        last_added = pixel

                # flag whether each corner has been crossed
                if i == len(group) - 1:
                    if corner == 0:
                        if coord[0] == 0:
                            corners[corner] = True
                    elif corner == 1:
                        if coord[1] == self.width - 1:
                            corners[corner] = True
                    elif corner == 2:
                        if coord[0] == self.height - 1:
                            corners[corner] = True
                    elif corner == 3:
                        if coord[1] == 0.0:
                            corners[corner] = True

                prev = coord
            dist = self._calculate_distance_from_top_left(sequence)

            sequences.append({"distance": dist, "sequence": sequence})

        # check whether we need to insert any missing corners
        if corners[0] is False:
            sequences.append({"distance": 0, "sequence": [(0, 0)]})
        if corners[1] is False:
            sequences.append({"distance": self.width, "sequence": [(0, self.width - 1)]})
        if corners[2] is False:
            sequences.append({"distance": self.width + self.height, "sequence": [(self.height - 1, self.width - 1)]})
        if corners[3] is False:
            sequences.append({"distance": 2 * self.width + self.height, "sequence": [(self.height - 1, 0)]})

        # join the sequences into a single contour
        # starting at top left and rotating clockwise
        sequences.sort(key=lambda x: x.get("distance"))  # type: ignore

        last = (-1, -1)
        for _sequence in sequences:
            if _sequence["sequence"][0] == last:  # type: ignore
                pixels.pop()
            if pixels:
                pixels = [*pixels, *_sequence["sequence"]]  # type: ignore
            else:
                pixels = _sequence["sequence"]  # type: ignore
            last = pixels[-1]

        if pixels[0] == last:
            pixels.pop(0)

        if pixels[0] == (0, 0):
            pixels.append(pixels.pop(0))

        return np.flip(convert_to_numpy(pixels, dtype=np.int32))  # type: ignore


class GenerateInstanceContour(Transform):
    """
    Generate contour for each instance in a 2D array. Use `GenerateSuccinctContour` to only include
    the pixels to which lines need to be drawn

    Args:
        min_num_points: assumed that the created contour does not form a contour if it does not contain more points
            than the specified value. Defaults to 3.
        contour_level: an optional value for `skimage.measure.find_contours` to find contours in the array.
            If not provided, the level is set to `(max(image) + min(image)) / 2`.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, min_num_points: int = 3, contour_level: float | None = None) -> None:
        self.contour_level = contour_level
        self.min_num_points = min_num_points

    def __call__(self, inst_mask: NdarrayOrTensor, offset: Sequence[int] | None = (0, 0)) -> np.ndarray | None:
        """
        Args:
            inst_mask: segmentation mask for a single instance. Shape should be [1, H, W, [D]]
            offset: optional offset of starting position of the instance mask in the original array. Default to 0 for each dim.
        """
        inst_mask = inst_mask.squeeze()  # squeeze channel dim
        inst_mask = convert_to_numpy(inst_mask)
        inst_contour_cv = find_contours(inst_mask, level=self.contour_level)
        generate_contour = GenerateSuccinctContour(inst_mask.shape[0], inst_mask.shape[1])
        inst_contour = generate_contour(inst_contour_cv)
        if inst_contour is None:
            return None
        # less than `self.min_num_points` points don't make a contour, so skip.
        # They are likely to be artifacts as the contours obtained via approximation.
        if inst_contour.shape[0] < self.min_num_points:
            print(f"< {self.min_num_points} points don't make a contour, so skipped!")
            return None
        # check for tricky shape
        elif len(inst_contour.shape) != 2:
            print(f"{len(inst_contour.shape)} != 2, check for tricky shapes!")
            return None
        else:
            inst_contour[:, 0] += offset[0]  # type: ignore
            inst_contour[:, 1] += offset[1]  # type: ignore
            return inst_contour


class GenerateInstanceCentroid(Transform):
    """
    Generate instance centroid using `skimage.measure.centroid`.

    Args:
        dtype: the data type of output centroid.

    """

    backend = [TransformBackends.NUMPY]

    def __init__(self, dtype: DtypeLike | None = int) -> None:
        self.dtype = dtype

    def __call__(self, inst_mask: NdarrayOrTensor, offset: Sequence[int] | int = 0) -> NdarrayOrTensor:
        """
        Args:
            inst_mask: segmentation mask for a single instance. Shape should be [1, H, W, [D]]
            offset: optional offset of starting position of the instance mask in the original array. Default to 0 for each dim.

        """
        inst_mask = convert_to_numpy(inst_mask)
        inst_mask = inst_mask.squeeze(0)  # squeeze channel dim
        ndim = len(inst_mask.shape)
        offset = ensure_tuple_rep(offset, ndim)

        inst_centroid = centroid(inst_mask)
        for i in range(ndim):
            inst_centroid[i] += offset[i]

        return convert_to_dst_type(inst_centroid, inst_mask, dtype=self.dtype)[0]


class GenerateInstanceType(Transform):
    """
    Generate instance type and probability for each instance.
    """

    backend = [TransformBackends.NUMPY]

    def __call__(  # type: ignore
        self, type_pred: NdarrayOrTensor, seg_pred: NdarrayOrTensor, bbox: np.ndarray, instance_id: int
    ) -> tuple[int, float]:
        """
        Args:
            type_pred: pixel-level type prediction map after activation function.
            seg_pred: pixel-level segmentation prediction map after activation function.
            bbox: bounding box coordinates of the instance, shape is [channel, 2 * spatial dims].
            instance_id: get instance type from specified instance id.
        """

        rmin, rmax, cmin, cmax = bbox.flatten()
        seg_map_crop = seg_pred[0, rmin:rmax, cmin:cmax]
        type_map_crop = type_pred[0, rmin:rmax, cmin:cmax]

        seg_map_crop = convert_to_dst_type(seg_map_crop == instance_id, type_map_crop, dtype=bool)[0]

        inst_type = type_map_crop[seg_map_crop]
        type_list, type_pixels = unique(inst_type, return_counts=True)
        type_list = list(zip(type_list, type_pixels))
        type_list = sorted(type_list, key=lambda x: x[1], reverse=True)
        inst_type = type_list[0][0]
        if inst_type == 0:  # ! pick the 2nd most dominant if exist
            if len(type_list) > 1:
                inst_type = type_list[1][0]
        type_dict = {v[0]: v[1] for v in type_list}
        type_prob = type_dict[inst_type] / (sum(seg_map_crop) + 1.0e-6)

        return (int(inst_type), float(type_prob))


class HoVerNetInstanceMapPostProcessing(Transform):
    """
    The post-processing transform for HoVerNet model to generate instance segmentation map.
    It generates an instance segmentation map as well as a dictionary containing centroids, bounding boxes, and contours
    for each instance.

    Args:
        activation: the activation layer to be applied on the input probability map.
            It can be "softmax" or "sigmoid" string, or any callable. Defaults to "softmax".
        mask_threshold: a float value to threshold to binarize probability map to generate mask.
        min_object_size: objects smaller than this size (in pixel) are removed. Defaults to 10.
        sobel_kernel_size: the size of the Sobel kernel used in :py:class:`GenerateInstanceBorder`. Defaults to 5.
        distance_smooth_fn: smoothing function for distance map.
            If not provided, :py:class:`monai.transforms.intensity.GaussianSmooth()` will be used.
        marker_threshold: a float value to threshold to binarize instance border map for markers.
            It turns uncertain area to 1 and other area to 0. Defaults to 0.4.
        marker_radius: the radius of the disk-shaped footprint used in `opening` of markers. Defaults to 2.
        marker_postprocess_fn: post-process function for watershed markers.
            If not provided, :py:class:`monai.transforms.post.FillHoles()` will be used.
        watershed_connectivity: `connectivity` argument of `skimage.segmentation.watershed`.
        min_num_points: minimum number of points to be considered as a contour. Defaults to 3.
        contour_level: an optional value for `skimage.measure.find_contours` to find contours in the array.
            If not provided, the level is set to `(max(image) + min(image)) / 2`.
        device: target device to put the output Tensor data.
    """

    def __init__(
        self,
        activation: str | Callable = "softmax",
        mask_threshold: float | None = None,
        min_object_size: int = 10,
        sobel_kernel_size: int = 5,
        distance_smooth_fn: Callable | None = None,
        marker_threshold: float = 0.4,
        marker_radius: int = 2,
        marker_postprocess_fn: Callable | None = None,
        watershed_connectivity: int | None = 1,
        min_num_points: int = 3,
        contour_level: float | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.generate_watershed_mask = GenerateWatershedMask(
            activation=activation, threshold=mask_threshold, min_object_size=min_object_size
        )
        self.generate_instance_border = GenerateInstanceBorder(kernel_size=sobel_kernel_size)
        self.generate_distance_map = GenerateDistanceMap(smooth_fn=distance_smooth_fn)
        self.generate_watershed_markers = GenerateWatershedMarkers(
            threshold=marker_threshold,
            radius=marker_radius,
            postprocess_fn=marker_postprocess_fn,
            min_object_size=min_object_size,
        )
        self.watershed = Watershed(connectivity=watershed_connectivity)
        self.generate_instance_contour = GenerateInstanceContour(
            min_num_points=min_num_points, contour_level=contour_level
        )
        self.generate_instance_centroid = GenerateInstanceCentroid()

    def __call__(  # type: ignore
        self, nuclear_prediction: NdarrayOrTensor, hover_map: NdarrayOrTensor
    ) -> tuple[dict, NdarrayOrTensor]:
        """post-process instance segmentation branches (NP and HV) to generate instance segmentation map.

        Args:
            nuclear_prediction: the output of NP (nuclear prediction) branch of HoVerNet model
            hover_map: the output of HV (hover map) branch of HoVerNet model
        """

        # Process NP and HV branch using watershed algorithm
        watershed_mask = self.generate_watershed_mask(nuclear_prediction)
        instance_borders = self.generate_instance_border(watershed_mask, hover_map)
        distance_map = self.generate_distance_map(watershed_mask, instance_borders)
        watershed_markers = self.generate_watershed_markers(watershed_mask, instance_borders)
        instance_map = self.watershed(distance_map, watershed_mask, watershed_markers)

        # Create bounding boxes, contours and centroids
        instance_ids = set(np.unique(instance_map)) - {0}  # exclude background
        instance_info = {}
        for inst_id in instance_ids:
            instance_mask = instance_map == inst_id
            instance_bbox = BoundingRect()(instance_mask)

            instance_mask = instance_mask[
                :, instance_bbox[0][0] : instance_bbox[0][1], instance_bbox[0][2] : instance_bbox[0][3]
            ]
            offset = [instance_bbox[0][2], instance_bbox[0][0]]
            instance_contour = self.generate_instance_contour(FillHoles()(instance_mask), offset)
            if instance_contour is not None:
                instance_centroid = self.generate_instance_centroid(instance_mask, offset)
                instance_info[inst_id] = {
                    "bounding_box": instance_bbox,
                    "centroid": instance_centroid,
                    "contour": instance_contour,
                }
        instance_map = convert_to_tensor(instance_map, device=self.device)
        return instance_info, instance_map


class HoVerNetNuclearTypePostProcessing(Transform):
    """
    The post-processing transform for HoVerNet model to generate nuclear type information.
    It updates the input instance info dictionary with information about types of the nuclei (value and probability).
    Also if requested (`return_type_map=True`), it generates a pixel-level type map.

    Args:
        activation: the activation layer to be applied on nuclear type branch. It can be "softmax" or "sigmoid" string,
            or any callable. Defaults to "softmax".
        threshold: an optional float value to threshold to binarize probability map.
            If not provided, defaults to 0.5 when activation is not "softmax", otherwise None.
        return_type_map: whether to calculate and return pixel-level type map.
        device: target device to put the output Tensor data.

    """

    def __init__(
        self,
        activation: str | Callable = "softmax",
        threshold: float | None = None,
        return_type_map: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.return_type_map = return_type_map
        self.generate_instance_type = GenerateInstanceType()

        # set activation layer
        use_softmax = False
        use_sigmoid = False
        activation_fn = None
        if isinstance(activation, str):
            if activation.lower() == "softmax":
                use_softmax = True
            elif activation.lower() == "sigmoid":
                use_sigmoid = True
            else:
                raise ValueError(
                    f"The activation should be 'softmax' or 'sigmoid' string, or any callable. '{activation}' was given."
                )
        elif callable(activation):
            activation_fn = activation
        else:
            raise ValueError(f"The activation type should be either str or callable. '{type(activation)}' was given.")
        self.activation = Activations(softmax=use_softmax, sigmoid=use_sigmoid, other=activation_fn)

        # set discretization transform
        if not use_softmax and threshold is None:
            threshold = 0.5
        self.as_discrete = AsDiscrete(threshold=threshold, argmax=use_softmax)

    def __call__(  # type: ignore
        self, type_prediction: NdarrayOrTensor, instance_info: dict[int, dict], instance_map: NdarrayOrTensor
    ) -> tuple[dict, NdarrayOrTensor | None]:
        """Process NC (type prediction) branch and combine it with instance segmentation
        It updates the instance_info with instance type and associated probability, and generate instance type map.

        Args:
            instance_info: instance information dictionary, the output of :py:class:`HoVerNetInstanceMapPostProcessing`
            instance_map: instance segmentation map, the output of :py:class:`HoVerNetInstanceMapPostProcessing`
            type_prediction: the output of NC (type prediction) branch of HoVerNet model
        """
        type_prediction = self.activation(type_prediction)
        type_prediction = self.as_discrete(type_prediction)

        type_map = None
        if self.return_type_map:
            type_map = convert_to_dst_type(torch.zeros(instance_map.shape), instance_map)[0]

        for inst_id in instance_info:
            instance_type, instance_type_prob = self.generate_instance_type(
                type_pred=type_prediction,
                seg_pred=instance_map,
                bbox=instance_info[inst_id]["bounding_box"],
                instance_id=inst_id,
            )
            # update instance info dict with type data
            instance_info[inst_id]["type_prob"] = instance_type_prob
            instance_info[inst_id]["type"] = instance_type

            # update instance type map
            if type_map is not None:
                type_map[instance_map == inst_id] = instance_type
                type_map = convert_to_tensor(type_map, device=self.device)

        return instance_info, type_map
