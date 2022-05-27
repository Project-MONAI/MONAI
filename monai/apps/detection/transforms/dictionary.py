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
"""
A collection of dictionary-based wrappers around the "vanilla" transforms for box operations
defined in :py:class:`monai.apps.detection.transforms.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from copy import deepcopy
from enum import Enum
from typing import Dict, Hashable, List, Mapping, Optional, Sequence, Type, Union

import numpy as np
import torch

from monai.apps.detection.transforms.array import (
    AffineBox,
    BoxMaskToBox,
    BoxToBoxMask,
    ClipBoxToImage,
    ConvertBoxMode,
    ConvertBoxToStandardMode,
    FlipBox,
    ZoomBox,
)
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import COMPUTE_DTYPE, BoxMode
from monai.data.utils import orientation_ras_lps
from monai.transforms import Flip, RandFlip, RandZoom, SpatialPad, Zoom
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.utils import InterpolateMode, NumpyPadMode, PytorchPadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import PostFix, TraceKeys
from monai.utils.type_conversion import convert_data_type

__all__ = [
    "ConvertBoxModed",
    "ConvertBoxModeD",
    "ConvertBoxModeDict",
    "ConvertBoxToStandardModed",
    "ConvertBoxToStandardModeD",
    "ConvertBoxToStandardModeDict",
    "AffineBoxToImageCoordinated",
    "AffineBoxToImageCoordinateD",
    "AffineBoxToImageCoordinateDict",
    "ZoomBoxd",
    "ZoomBoxD",
    "ZoomBoxDict",
    "RandZoomBoxd",
    "RandZoomBoxD",
    "RandZoomBoxDict",
    "FlipBoxd",
    "FlipBoxD",
    "FlipBoxDict",
    "RandFlipBoxd",
    "RandFlipBoxD",
    "RandFlipBoxDict",
    "ClipBoxToImaged",
    "ClipBoxToImageD",
    "ClipBoxToImageDict",
    "BoxToBoxMaskd",
    "BoxToBoxMaskD",
    "BoxToBoxMaskDict",
    "BoxMaskToBoxd",
    "BoxMaskToBoxD",
    "BoxMaskToBoxDict",
]

DEFAULT_POST_FIX = PostFix.meta()
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
PadModeSequence = Union[Sequence[Union[NumpyPadMode, PytorchPadMode, str]], NumpyPadMode, PytorchPadMode, str]


class ConvertBoxModed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.ConvertBoxMode`.

    This transform converts the boxes in src_mode to the dst_mode.

    Example:
        .. code-block:: python

            data = {"boxes": torch.ones(10,4)}
            # convert boxes with format [xmin, ymin, xmax, ymax] to [xcenter, ycenter, xsize, ysize].
            box_converter = ConvertBoxModed(box_keys=["boxes"], src_mode="xyxy", dst_mode="ccwh")
            box_converter(data)
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        src_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
        dst_mode: Union[str, BoxMode, Type[BoxMode], None] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            box_keys: Keys to pick data for transformation.
            src_mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .
            dst_mode: target box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .
            allow_missing_keys: don't raise exception if key is missing.

        See also :py:class:`monai.apps.detection,transforms.array.ConvertBoxMode`
        """
        super().__init__(box_keys, allow_missing_keys)
        self.converter = ConvertBoxMode(src_mode=src_mode, dst_mode=dst_mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key, extra_info={"src": self.converter.src_mode, "dst": self.converter.dst_mode})
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            tr = self.get_most_recent_transform(d, key)
            src_mode, dst_mode = tr[TraceKeys.EXTRA_INFO]["src"], tr[TraceKeys.EXTRA_INFO]["dst"]
            inverse_converter = ConvertBoxMode(src_mode=dst_mode, dst_mode=src_mode)
            # Inverse is same as forward
            d[key] = inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class ConvertBoxToStandardModed(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.ConvertBoxToStandardMode`.

    Convert given boxes to standard mode.
    Standard mode is "xyxy" or "xyzxyz",
    representing box format of [xmin, ymin, xmax, ymax] or [xmin, ymin, zmin, xmax, ymax, zmax].

    Example:
        .. code-block:: python

            data = {"boxes": torch.ones(10,6)}
            # convert boxes with format [xmin, xmax, ymin, ymax, zmin, zmax] to [xmin, ymin, zmin, xmax, ymax, zmax]
            box_converter = ConvertBoxToStandardModed(box_keys=["boxes"], mode="xxyyzz")
            box_converter(data)
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        mode: Union[str, BoxMode, Type[BoxMode], None] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            box_keys: Keys to pick data for transformation.
            mode: source box mode. If it is not given, this func will assume it is ``StandardMode()``.
                It follows the same format with ``src_mode`` in :class:`~monai.apps.detection.transforms.array.ConvertBoxMode` .
            allow_missing_keys: don't raise exception if key is missing.

        See also :py:class:`monai.apps.detection,transforms.array.ConvertBoxToStandardMode`
        """
        super().__init__(box_keys, allow_missing_keys)
        self.converter = ConvertBoxToStandardMode(mode=mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key, extra_info={"mode": self.converter.mode})
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            tr = self.get_most_recent_transform(d, key)
            original_mode = tr[TraceKeys.EXTRA_INFO]["mode"]
            inverse_converter = ConvertBoxMode(src_mode=None, dst_mode=original_mode)
            # Inverse is same as forward
            d[key] = inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class AffineBoxToImageCoordinated(MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that converts box in world coordinate to image coordinate.

    Args:
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: The single key that represents the reference image to which ``box_keys`` are attached.
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.
        image_meta_key: explicitly indicate the key of the corresponding metadata dictionary.
            for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
            the metadata is a dictionary object which contains: filename, affine, original_shape, etc.
            it is a string, map to the `box_ref_image_key`.
            if None, will try to construct meta_keys by `box_ref_image_key_{meta_key_postfix}`.
        image_meta_key_postfix: if image_meta_keys=None, use `box_ref_image_key_{postfix}` to fetch the metadata according
            to the key data, default is `meta_dict`, the metadata is a dictionary object.
            For example, to handle key `image`,  read/write affine matrices from the
            metadata `image_meta_dict` dictionary's `affine` field.
        affine_lps_to_ras: default ``False``. Yet if 1) the image is read by ITKReader,
            and 2) the ITKReader has affine_lps_to_ras=True, and 3) the box is in world coordinate,
            then set ``affine_lps_to_ras=True``.
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_ref_image_keys: str,
        allow_missing_keys: bool = False,
        image_meta_key: Union[str, None] = None,
        image_meta_key_postfix: Union[str, None] = DEFAULT_POST_FIX,
        affine_lps_to_ras=False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        box_ref_image_keys_tuple = ensure_tuple(box_ref_image_keys)
        if len(box_ref_image_keys_tuple) > 1:
            raise ValueError(
                "Please provide a single key for box_ref_image_keys.\
                All boxes of box_keys are attached to box_ref_image_keys."
            )
        self.image_meta_key = image_meta_key or f"{box_ref_image_keys}_{image_meta_key_postfix}"
        self.converter_to_image_coordinate = AffineBox()
        self.affine_lps_to_ras = affine_lps_to_ras

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        meta_key = self.image_meta_key
        # extract affine matrix from meta_data
        if meta_key not in d:
            raise ValueError(f"{meta_key} is not found. Please check whether it is the correct the image meta key.")
        if "affine" not in d[meta_key]:
            raise ValueError(
                f"'affine' is not found in {meta_key}. \
                Please check whether it is the correct the image meta key."
            )
        affine: NdarrayOrTensor = d[meta_key]["affine"]  # type: ignore
        if self.affine_lps_to_ras:  # RAS affine
            affine = orientation_ras_lps(affine)

        # when convert boxes from world coordinate to image coordinate,
        # we apply inverse affine transform
        affine_t, *_ = convert_data_type(affine, torch.Tensor)
        # torch.inverse should not run in half precision
        inv_affine_t = torch.inverse(affine_t.to(COMPUTE_DTYPE))

        for key in self.key_iterator(d):
            self.push_transform(d, key, extra_info={"affine": affine})
            d[key] = self.converter_to_image_coordinate(d[key], affine=inv_affine_t)
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            affine = transform["extra_info"]["affine"]
            d[key] = AffineBox()(d[key], affine=affine)
            self.pop_transform(d, key)
        return d


class ZoomBoxd(MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that zooms input boxes and images with the given zoom scale.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represents the reference images to which ``box_keys`` are attached.
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        kwargs: other arguments for the `np.pad` or `torch.pad` function.
            note that `np.pad` treats channel dimension as the first dimension.
    """

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        zoom: Union[Sequence[float], float],
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        padding_mode: PadModeSequence = NumpyPadMode.EDGE,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        super().__init__(self.image_keys + self.box_keys, allow_missing_keys)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

        self.mode = ensure_tuple_rep(mode, len(self.image_keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.image_keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.image_keys))
        self.zoomer = Zoom(zoom=zoom, keep_size=keep_size, **kwargs)
        self.keep_size = keep_size

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        # zoom box
        for box_key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            src_spatial_size = d[box_ref_image_key].shape[1:]
            dst_spatial_size = [int(round(z * ss)) for z, ss in zip(self.zoomer.zoom, src_spatial_size)]  # type: ignore
            self.zoomer.zoom = [ds / float(ss) for ss, ds in zip(src_spatial_size, dst_spatial_size)]
            self.push_transform(
                d,
                box_key,
                extra_info={"zoom": self.zoomer.zoom, "src_spatial_size": src_spatial_size, "type": "box_key"},
            )
            d[box_key] = ZoomBox(zoom=self.zoomer.zoom, keep_size=self.keep_size)(
                d[box_key], src_spatial_size=src_spatial_size
            )

        # zoom image, copied from monai.transforms.spatial.dictionary.Zoomd
        for key, mode, padding_mode, align_corners in zip(
            self.image_keys, self.mode, self.padding_mode, self.align_corners
        ):
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                    "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                    "original_shape": d[key].shape[1:],
                    "type": "image_key",
                },
            )
            d[key] = self.zoomer(d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners)

        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            key_type = transform[TraceKeys.EXTRA_INFO]["type"]
            # zoom image, copied from monai.transforms.spatial.dictionary.Zoomd
            if key_type == "image_key":
                # Create inverse transform
                zoom = np.array(self.zoomer.zoom)
                inverse_transform = Zoom(zoom=(1 / zoom).tolist(), keep_size=self.zoomer.keep_size)
                mode = transform[TraceKeys.EXTRA_INFO]["mode"]
                padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
                align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
                # Apply inverse
                d[key] = inverse_transform(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=None if align_corners == TraceKeys.NONE else align_corners,
                )
                # Size might be out by 1 voxel so pad
                d[key] = SpatialPad(transform[TraceKeys.EXTRA_INFO]["original_shape"], mode="edge")(d[key])

            # zoom boxes
            if key_type == "box_key":
                zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                src_spatial_size = transform[TraceKeys.EXTRA_INFO]["src_spatial_size"]
                box_inverse_transform = ZoomBox(zoom=(1 / zoom).tolist(), keep_size=self.zoomer.keep_size)
                d[key] = box_inverse_transform(d[key], src_spatial_size=src_spatial_size)

            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandZoomBoxd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that randomly zooms input boxes and images with given probability within given zoom range.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represents the reference images to which ``box_keys`` are attached.
        prob: Probability of zooming.
        min_zoom: Min zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, min_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        max_zoom: Max zoom factor. Can be float or sequence same size as image.
            If a float, select a random factor from `[min_zoom, max_zoom]` then apply to all spatial dims
            to keep the original spatial shape ratio.
            If a sequence, max_zoom should contain one value for each spatial axis.
            If 2 values provided for 3D data, use the first value for both H & W dims to keep the same zoom ratio.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: available modes for numpy array:{``"constant"``, ``"edge"``, ``"linear_ramp"``, ``"maximum"``,
            ``"mean"``, ``"median"``, ``"minimum"``, ``"reflect"``, ``"symmetric"``, ``"wrap"``, ``"empty"``}
            available modes for PyTorch Tensor: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}.
            One of the listed string values or a user supplied function. Defaults to ``"constant"``.
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
        kwargs: other args for `np.pad` API, note that `np.pad` treats channel dimension as the first dimension.
            more details: https://numpy.org/doc/1.18/reference/generated/numpy.pad.html
    """

    backend = RandZoom.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        prob: float = 0.1,
        min_zoom: Union[Sequence[float], float] = 0.9,
        max_zoom: Union[Sequence[float], float] = 1.1,
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        padding_mode: PadModeSequence = NumpyPadMode.EDGE,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
        **kwargs,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        MapTransform.__init__(self, self.image_keys + self.box_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

        self.rand_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, keep_size=keep_size, **kwargs)
        self.mode = ensure_tuple_rep(mode, len(self.image_keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.image_keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.image_keys))
        self.keep_size = keep_size

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandZoomBoxd":
        super().set_random_state(seed, state)
        self.rand_zoom.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        first_key: Union[Hashable, List] = self.first_key(d)
        if first_key == []:
            return d

        self.randomize(None)

        # all the keys share the same random zoom factor
        self.rand_zoom.randomize(d[first_key])  # type: ignore

        # zoom box
        for box_key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            if self._do_transform:
                src_spatial_size = d[box_ref_image_key].shape[1:]
                dst_spatial_size = [int(round(z * ss)) for z, ss in zip(self.rand_zoom._zoom, src_spatial_size)]
                self.rand_zoom._zoom = [ds / float(ss) for ss, ds in zip(src_spatial_size, dst_spatial_size)]

                self.push_transform(
                    d,
                    box_key,
                    extra_info={"zoom": self.rand_zoom._zoom, "src_spatial_size": src_spatial_size, "type": "box_key"},
                )
                d[box_key] = ZoomBox(zoom=self.rand_zoom._zoom, keep_size=self.keep_size)(
                    d[box_key], src_spatial_size=src_spatial_size
                )

        # zoom image, copied from monai.transforms.spatial.dictionary.RandZoomd
        for key, mode, padding_mode, align_corners in zip(
            self.image_keys, self.mode, self.padding_mode, self.align_corners
        ):
            if self._do_transform:
                self.push_transform(
                    d,
                    key,
                    extra_info={
                        "zoom": self.rand_zoom._zoom,
                        "mode": mode.value if isinstance(mode, Enum) else mode,
                        "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                        "original_shape": d[key].shape[1:],
                        "type": "image_key",
                    },
                )
                d[key] = self.rand_zoom(
                    d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners, randomize=False
                )

        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            key_type = transform[TraceKeys.EXTRA_INFO]["type"]
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # zoom image, copied from monai.transforms.spatial.dictionary.Zoomd
                if key_type == "image_key":
                    zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                    mode = transform[TraceKeys.EXTRA_INFO]["mode"]
                    padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
                    align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
                    inverse_transform = Zoom(zoom=(1.0 / zoom).tolist(), keep_size=self.rand_zoom.keep_size)
                    d[key] = inverse_transform(
                        d[key],
                        mode=mode,
                        padding_mode=padding_mode,
                        align_corners=None if align_corners == TraceKeys.NONE else align_corners,
                    )
                    # Size might be out by 1 voxel so pad
                    d[key] = SpatialPad(transform[TraceKeys.EXTRA_INFO]["original_shape"], mode="edge")(d[key])

                # zoom boxes
                if key_type == "box_key":
                    # Create inverse transform
                    zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                    src_spatial_size = transform[TraceKeys.EXTRA_INFO]["src_spatial_size"]
                    box_inverse_transform = ZoomBox(zoom=(1.0 / zoom).tolist(), keep_size=self.rand_zoom.keep_size)
                    d[key] = box_inverse_transform(d[key], src_spatial_size=src_spatial_size)

                # Remove the applied transform
                self.pop_transform(d, key)
        return d


class FlipBoxd(MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that flip boxes and images.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represents the reference images to which ``box_keys`` are attached.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = Flip.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        super().__init__(self.image_keys + self.box_keys, allow_missing_keys)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

        self.flipper = Flip(spatial_axis=spatial_axis)
        self.box_flipper = FlipBox(spatial_axis=self.flipper.spatial_axis)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for key in self.image_keys:
            d[key] = self.flipper(d[key])
            self.push_transform(d, key, extra_info={"type": "image_key"})

        for box_key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            spatial_size = d[box_ref_image_key].shape[1:]
            d[box_key] = self.box_flipper(d[box_key], spatial_size)
            self.push_transform(d, box_key, extra_info={"spatial_size": spatial_size, "type": "box_key"})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            key_type = transform[TraceKeys.EXTRA_INFO]["type"]

            # flip image, copied from monai.transforms.spatial.dictionary.Flipd
            if key_type == "image_key":
                d[key] = self.flipper(d[key])

            # flip boxes
            if key_type == "box_key":
                spatial_size = transform[TraceKeys.EXTRA_INFO]["spatial_size"]
                d[key] = self.box_flipper(d[key], spatial_size)

            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class RandFlipBoxd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based transform that randomly flip boxes and images with the given probabilities.

    Args:
        image_keys: Keys to pick image data for transformation.
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_ref_image_keys: Keys that represents the reference images to which ``box_keys`` are attached.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    backend = RandFlip.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        box_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        self.image_keys = ensure_tuple(image_keys)
        self.box_keys = ensure_tuple(box_keys)
        MapTransform.__init__(self, self.image_keys + self.box_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))

        self.flipper = RandFlip(prob=1.0, spatial_axis=spatial_axis)
        self.box_flipper = FlipBox(spatial_axis=spatial_axis)

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandFlipBoxd":
        super().set_random_state(seed, state)
        self.flipper.set_random_state(seed, state)
        return self

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize(None)

        for key in self.image_keys:
            if self._do_transform:
                d[key] = self.flipper(d[key], randomize=False)
            self.push_transform(d, key, extra_info={"type": "image_key"})

        for box_key, box_ref_image_key in zip(self.box_keys, self.box_ref_image_keys):
            spatial_size = d[box_ref_image_key].shape[1:]
            if self._do_transform:
                d[box_key] = self.box_flipper(d[box_key], spatial_size)
            self.push_transform(d, box_key, extra_info={"spatial_size": spatial_size, "type": "box_key"})
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            key_type = transform[TraceKeys.EXTRA_INFO]["type"]
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # flip image, copied from monai.transforms.spatial.dictionary.RandFlipd
                if key_type == "image_key":
                    d[key] = self.flipper(d[key], randomize=False)

                # flip boxes
                if key_type == "box_key":
                    spatial_size = transform[TraceKeys.EXTRA_INFO]["spatial_size"]
                    d[key] = self.box_flipper(d[key], spatial_size)

            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class ClipBoxToImaged(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.ClipBoxToImage`.

    Clip the bounding boxes and the associated labels/scores to makes sure they are within the image.
    There might be multiple keys of labels/scores associated with one key of boxes.

    Args:
        box_keys: The single key to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        label_keys: Keys that represents the lables corresponding to the ``box_keys``. Multiple keys are allowed.
        box_ref_image_keys: The single key that represents the reference image
            to which ``box_keys`` and ``label_keys`` are attached.
        remove_empty: whether to remove the boxes that are actually empty
        allow_missing_keys: don't raise exception if key is missing.

    Example:
        .. code-block:: python

            ClipBoxToImaged(
                box_keys="boxes", box_ref_image_keys="image", label_keys=["labels", "scores"], remove_empty=True
            )
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        label_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        remove_empty: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        box_keys_tuple = ensure_tuple(box_keys)
        if len(box_keys_tuple) != 1:
            raise ValueError(
                "Please provide a single key for box_keys.\
                All label_keys are attached to this box_keys."
            )
        box_ref_image_keys_tuple = ensure_tuple(box_ref_image_keys)
        if len(box_ref_image_keys_tuple) != 1:
            raise ValueError(
                "Please provide a single key for box_ref_image_keys.\
                All box_keys and label_keys are attached to this box_ref_image_keys."
            )
        self.label_keys = ensure_tuple(label_keys)
        super().__init__(box_keys_tuple, allow_missing_keys)

        self.box_keys = box_keys_tuple[0]
        self.box_ref_image_keys = box_ref_image_keys_tuple[0]
        self.clipper = ClipBoxToImage(remove_empty=remove_empty)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        spatial_size = d[self.box_ref_image_keys].shape[1:]
        labels = [d[label_key] for label_key in self.label_keys]  # could be multiple arrays
        d[self.box_keys], clipped_labels = self.clipper(d[self.box_keys], labels, spatial_size)

        for label_key, clipped_labels_i in zip(self.label_keys, clipped_labels):
            d[label_key] = clipped_labels_i
        return d


class BoxToBoxMaskd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.BoxToBoxMask`.
    Pairs with :py:class:`monai.apps.detection.transforms.dictionary.BoxMaskToBoxd` .
    Please make sure the same ``min_fg_label`` is used when using the two transforms in pairs.
    The output d[box_mask_key] will have background intensity 0, since the following operations may pad 0 on the border.

    This is the general solution for transforms that need to be applied on images and boxes simultaneously.
    It is performed with the following steps.
    1) use BoxToBoxMaskd to covert boxes and labels to box_masks;
    2) do transforms, e.g., rotation or cropping, on images and box_masks together;
    3) use BoxMaskToBoxd to convert box_masks back to boxes and labels.

    Args:
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_mask_keys: Keys to store output box mask results for transformation. Same length with ``box_keys``.
        label_keys: Keys that represents the lables corresponding to the ``box_keys``. Same length with ``box_keys``.
        box_ref_image_keys: Keys that represents the reference images to which ``box_keys`` are attached.
        min_fg_label: min foreground box label.
        ellipse_mask: bool.
            If True, it assumes the object shape is close to ellipse or ellipsoid.
            If False, it assumes the object shape is close to rectangle or cube and well occupies the bounding box.
            If the users are going to apply random rotation as data augmentation, we suggest setting ellipse_mask=True
            See also Kalra et al. "Towards Rotation Invariance in Object Detection", ICCV 2021.
        allow_missing_keys: don't raise exception if key is missing.

    Example:
        .. code-block:: python

            # This code snippet creates transforms (random rotation and croppping) on boxes, labels, and image together.
            import numpy as np
            from monai.transforms import Compose, RandRotated, RandSpatialCropd, DeleteItemsd
            transforms = Compose(
                [
                    BoxToBoxMaskd(
                        box_keys="boxes", label_keys="labels",
                        box_mask_keys="box_mask", box_ref_image_keys="image",
                        min_fg_label=0, ellipse_mask=True
                    ),
                    RandRotated(keys=["image","box_mask"],mode=["nearest","nearest"],
                        prob=0.2,range_x=np.pi/6,range_y=np.pi/6,range_z=np.pi/6,
                        keep_size=True,padding_mode="zeros"
                    ),
                    RandSpatialCropd(keys=["image","box_mask"],roi_size=128, random_size=False),
                    BoxMaskToBoxd(
                        box_mask_keys="box_mask", box_keys="boxes",
                        label_keys="labels", min_fg_label=0
                    )
                    DeleteItemsd(keys=["box_mask"]),
                ]
            )

    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_mask_keys: KeysCollection,
        label_keys: KeysCollection,
        box_ref_image_keys: KeysCollection,
        min_fg_label: int,
        ellipse_mask: bool = False,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.box_keys = ensure_tuple(box_keys)
        self.label_keys = ensure_tuple(label_keys)
        self.box_mask_keys = ensure_tuple(box_mask_keys)
        if not len(self.label_keys) == len(self.box_keys) == len(self.box_mask_keys):
            raise ValueError("Please make sure len(label_keys)==len(box_keys)==len(box_mask_keys)!")
        self.box_ref_image_keys = ensure_tuple_rep(box_ref_image_keys, len(self.box_keys))
        self.bg_label = min_fg_label - 1  # make sure background label is always smaller than fg labels.
        self.converter = BoxToBoxMask(bg_label=self.bg_label, ellipse_mask=ellipse_mask)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for box_key, label_key, box_mask_key, box_ref_image_key in zip(
            self.box_keys, self.label_keys, self.box_mask_keys, self.box_ref_image_keys
        ):
            spatial_size = d[box_ref_image_key].shape[1:]
            d[box_mask_key] = self.converter(d[box_key], d[label_key], spatial_size)
            # make box mask background intensity to be 0, since the following operations may pad 0 on the border.
            d[box_mask_key] -= self.bg_label
        return d


class BoxMaskToBoxd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.apps.detection.transforms.array.BoxMaskToBox`.
    Pairs with :py:class:`monai.apps.detection.transforms.dictionary.BoxToBoxMaskd` .
    Please make sure the same ``min_fg_label`` is used when using the two transforms in pairs.

    This is the general solution for transforms that need to be applied on images and boxes simultaneously.
    It is performed with the following steps.
    1) use BoxToBoxMaskd to covert boxes and labels to box_masks;
    2) do transforms, e.g., rotation or cropping, on images and box_masks together;
    3) use BoxMaskToBoxd to convert box_masks back to boxes and labels.

    Args:
        box_keys: Keys to pick box data for transformation. The box mode is assumed to be ``StandardMode``.
        box_mask_keys: Keys to store output box mask results for transformation. Same length with ``box_keys``.
        label_keys: Keys that represents the lables corresponding to the ``box_keys``. Same length with ``box_keys``.
        min_fg_label: min foreground box label.
        box_dtype: output dtype for box_keys
        label_dtype: output dtype for label_keys
        allow_missing_keys: don't raise exception if key is missing.

    Example:
        .. code-block:: python

            # This code snippet creates transforms (random rotation and croppping) on boxes, labels, and images together.
            import numpy as np
            from monai.transforms import Compose, RandRotated, RandSpatialCropd, DeleteItemsd
            transforms = Compose(
                [
                    BoxToBoxMaskd(
                        box_keys="boxes", label_keys="labels",
                        box_mask_keys="box_mask", box_ref_image_keys="image",
                        min_fg_label=0, ellipse_mask=True
                    ),
                    RandRotated(keys=["image","box_mask"],mode=["nearest","nearest"],
                        prob=0.2,range_x=np.pi/6,range_y=np.pi/6,range_z=np.pi/6,
                        keep_size=True,padding_mode="zeros"
                    ),
                    RandSpatialCropd(keys=["image","box_mask"],roi_size=128, random_size=False),
                    BoxMaskToBoxd(
                        box_mask_keys="box_mask", box_keys="boxes",
                        label_keys="labels", min_fg_label=0
                    )
                    DeleteItemsd(keys=["box_mask"]),
                ]
            )
    """

    def __init__(
        self,
        box_keys: KeysCollection,
        box_mask_keys: KeysCollection,
        label_keys: KeysCollection,
        min_fg_label: int,
        box_dtype=torch.float32,
        label_dtype=torch.long,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(box_keys, allow_missing_keys)
        self.box_keys = ensure_tuple(box_keys)
        self.label_keys = ensure_tuple(label_keys)
        self.box_mask_keys = ensure_tuple(box_mask_keys)
        if not len(self.label_keys) == len(self.box_keys) == len(self.box_mask_keys):
            raise ValueError("Please make sure len(label_keys)==len(box_keys)==len(box_mask_keys)!")
        self.bg_label = min_fg_label - 1  # make sure background label is always smaller than fg labels.
        self.converter = BoxMaskToBox(bg_label=self.bg_label, box_dtype=box_dtype, label_dtype=label_dtype)
        self.box_dtype = box_dtype

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        for box_key, label_key, box_mask_key in zip(self.box_keys, self.label_keys, self.box_mask_keys):
            d[box_mask_key] += self.bg_label  # pairs with the operation in BoxToBoxMaskd
            d[box_key], d[label_key] = self.converter(d[box_mask_key])
        return d


ConvertBoxModeD = ConvertBoxModeDict = ConvertBoxModed
ConvertBoxToStandardModeD = ConvertBoxToStandardModeDict = ConvertBoxToStandardModed
ZoomBoxD = ZoomBoxDict = ZoomBoxd
RandZoomBoxD = RandZoomBoxDict = RandZoomBoxd
AffineBoxToImageCoordinateD = AffineBoxToImageCoordinateDict = AffineBoxToImageCoordinated
FlipBoxD = FlipBoxDict = FlipBoxd
RandFlipBoxD = RandFlipBoxDict = RandFlipBoxd
ClipBoxToImageD = ClipBoxToImageDict = ClipBoxToImaged
BoxToBoxMaskD = BoxToBoxMaskDict = BoxToBoxMaskd
BoxMaskToBoxD = BoxMaskToBoxDict = BoxMaskToBoxd
