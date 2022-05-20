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
from typing import Dict, Hashable, Mapping, Type, Union

from monai.apps.detection.transforms.array import ConvertBoxMode, ConvertBoxToStandardMode, ZoomBox
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.box_utils import BoxMode
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.transform import MapTransform
from monai.transforms.spatial.array import (
    RandZoom,
    )

__all__ = [
    "ConvertBoxModed",
    "ConvertBoxModeD",
    "ConvertBoxModeDict",
    "ConvertBoxToStandardModed",
    "ConvertBoxToStandardModeD",
    "ConvertBoxToStandardModeDict",
    "RandZoomBoxd",
    "RandZoomBoxD",
    "RandZoomBoxDict",
]


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
        self.inverse_converter = ConvertBoxMode(src_mode=dst_mode, dst_mode=src_mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Inverse is same as forward
            d[key] = self.inverse_converter(d[key])
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
        self.inverse_converter = ConvertBoxMode(src_mode=None, dst_mode=mode)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.converter(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Inverse is same as forward
            d[key] = self.inverse_converter(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d

class RandZoomBoxd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Randomly zooms input arrays with given probability within given zoom range.

    Args:
        box_keys: Keys to pick box data for transformation.
        image_keys: Keys to pick box data for transformation.
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
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
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
        box_keys: KeysCollection,
        image_keys: KeysCollection,
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
        MapTransform.__init__(self, image_keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_zoom = RandZoom(prob=1.0, min_zoom=min_zoom, max_zoom=max_zoom, keep_size=keep_size, **kwargs)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.box_keys = box_keys
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
        spatial_dims = len(d[first_key].shape)-1
        for key, mode, padding_mode, align_corners in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners
        ):
            if self._do_transform:
                orig_image_size = d[key].shape[1:]
                d[key] = self.rand_zoom(
                    d[key], mode=mode, padding_mode=padding_mode, align_corners=align_corners, randomize=False
                )

            self.push_transform(
                d,
                key,
                extra_info={
                    "zoom": self.rand_zoom._zoom,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                },
            )

        for box_key in self.box_keys:
            if self._do_transform:
                d[box_key] = ZoomBox(zoom=self.rand_zoom._zoom,keep_size=self.keep_size)(d[box_key],orig_image_size=orig_image_size)
                self.push_transform(
                    d,
                    box_key,
                    extra_info={
                        "zoom": self.rand_zoom._zoom,
                    },
                )

        return d

    def inverse(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Create inverse transform
                zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                mode = transform[TraceKeys.EXTRA_INFO]["mode"]
                padding_mode = transform[TraceKeys.EXTRA_INFO]["padding_mode"]
                align_corners = transform[TraceKeys.EXTRA_INFO]["align_corners"]
                inverse_transform = Zoom(zoom=(1 / zoom).tolist(), keep_size=self.rand_zoom.keep_size)
                # Apply inverse
                orig_image_size = d[key].shape[1:]
                d[key] = inverse_transform(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=None if align_corners == TraceKeys.NONE else align_corners,
                )
                # Size might be out by 1 voxel so pad
                d[key] = SpatialPad(transform[TraceKeys.ORIG_SIZE], mode="edge")(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        for box_key in self.box_keys:
            transform = self.get_most_recent_transform(d, box_key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[TraceKeys.DO_TRANSFORM]:
                # Create inverse transform
                zoom = np.array(transform[TraceKeys.EXTRA_INFO]["zoom"])
                box_inverse_transform = ZoomBox(zoom=(1 / zoom).tolist(), keep_size=self.rand_zoom.keep_size)
                d[box_key] = box_inverse_transform(
                    d[box_key], orig_image_size=orig_image_size
                )
            # Remove the applied transform
            self.pop_transform(d, box_key)

        return

ConvertBoxModeD = ConvertBoxModeDict = ConvertBoxModed
ConvertBoxToStandardModeD = ConvertBoxToStandardModeDict = ConvertBoxToStandardModed
