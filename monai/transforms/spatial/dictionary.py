# Copyright 2020 - 2021 MONAI Consortium
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
A collection of dictionary-based wrappers around the "vanilla" transforms for spatial operations
defined in :py:class:`monai.transforms.spatial.array`.

Class names are ended with 'd' to denote dictionary-based transforms.
"""

from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.networks.layers import AffineTransform
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.spatial.array import (
    AddCoordinateChannels,
    Affine,
    AffineGrid,
    Flip,
    Orientation,
    Rand2DElastic,
    Rand3DElastic,
    RandAffine,
    Resize,
    Rotate,
    Rotate90,
    Spacing,
    Zoom,
)
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
)
from monai.utils.enums import InverseKeys
from monai.utils.module import optional_import

nib, _ = optional_import("nibabel")

__all__ = [
    "Spacingd",
    "Orientationd",
    "Rotate90d",
    "RandRotate90d",
    "Resized",
    "Affined",
    "RandAffined",
    "Rand2DElasticd",
    "Rand3DElasticd",
    "Flipd",
    "RandFlipd",
    "RandAxisFlipd",
    "Rotated",
    "RandRotated",
    "Zoomd",
    "RandZoomd",
    "SpacingD",
    "SpacingDict",
    "OrientationD",
    "OrientationDict",
    "Rotate90D",
    "Rotate90Dict",
    "RandRotate90D",
    "RandRotate90Dict",
    "ResizeD",
    "ResizeDict",
    "AffineD",
    "AffineDict",
    "RandAffineD",
    "RandAffineDict",
    "Rand2DElasticD",
    "Rand2DElasticDict",
    "Rand3DElasticD",
    "Rand3DElasticDict",
    "FlipD",
    "FlipDict",
    "RandFlipD",
    "RandFlipDict",
    "RandAxisFlipD",
    "RandAxisFlipDict",
    "RotateD",
    "RotateDict",
    "RandRotateD",
    "RandRotateDict",
    "ZoomD",
    "ZoomDict",
    "RandZoomD",
    "RandZoomDict",
    "AddCoordinateChannelsD",
    "AddCoordinateChannelsDict",
]

GridSampleModeSequence = Union[Sequence[Union[GridSampleMode, str]], GridSampleMode, str]
GridSamplePadModeSequence = Union[Sequence[Union[GridSamplePadMode, str]], GridSamplePadMode, str]
InterpolateModeSequence = Union[Sequence[Union[InterpolateMode, str]], InterpolateMode, str]
NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]


class Spacingd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Spacing`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After resampling the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.

    see also:
        :py:class:`monai.transforms.Spacing`
    """

    def __init__(
        self,
        keys: KeysCollection,
        pixdim: Sequence[float],
        diagonal: bool = False,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
        align_corners: Union[Sequence[bool], bool] = False,
        dtype: Optional[Union[Sequence[DtypeLike], DtypeLike]] = np.float64,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            pixdim: output voxel spacing.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, pixdim_2, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, the axes orientation, orthogonal rotation and
                translations components from the original affine will be
                preserved in the target affine. This option will not flip/swap
                axes against the original ones.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of bool, each element corresponds to a key in ``keys``.
            dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
                If None, use the data type of input data. To be compatible with other modules,
                the output data type is always ``np.float32``.
                It also can be a sequence of dtypes, each element corresponds to a key in ``keys``.
            meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the meta data is a dictionary object which contains: filename, affine, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys=None, use `key_{postfix}` to to fetch the meta data according
                to the key data, default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        """
        super().__init__(keys, allow_missing_keys)
        self.spacing_transform = Spacing(pixdim, diagonal=diagonal)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        d: Dict = dict(data)
        for key, mode, padding_mode, align_corners, dtype, meta_key, meta_key_postfix in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype, self.meta_keys, self.meta_key_postfix
        ):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            original_spatial_shape = d[key].shape[1:]
            d[key], old_affine, new_affine = self.spacing_transform(
                data_array=np.asarray(d[key]),
                affine=meta_data["affine"],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
            )
            self.push_transform(
                d,
                key,
                extra_info={
                    "meta_key": meta_key,
                    "old_affine": old_affine,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                    "align_corners": align_corners if align_corners is not None else "none",
                },
                orig_size=original_spatial_shape,
            )
            # set the 'affine' key
            meta_data["affine"] = new_affine
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key, dtype in self.key_iterator(d, self.dtype):
            transform = self.get_most_recent_transform(d, key)
            if self.spacing_transform.diagonal:
                raise RuntimeError(
                    "Spacingd:inverse not yet implemented for diagonal=True. "
                    + "Please raise a github issue if you need this feature"
                )
            # Create inverse transform
            meta_data = d[transform[InverseKeys.EXTRA_INFO]["meta_key"]]
            old_affine = np.array(transform[InverseKeys.EXTRA_INFO]["old_affine"])
            mode = transform[InverseKeys.EXTRA_INFO]["mode"]
            padding_mode = transform[InverseKeys.EXTRA_INFO]["padding_mode"]
            align_corners = transform[InverseKeys.EXTRA_INFO]["align_corners"]
            orig_size = transform[InverseKeys.ORIG_SIZE]
            orig_pixdim = np.sqrt(np.sum(np.square(old_affine), 0))[:-1]
            inverse_transform = Spacing(orig_pixdim, diagonal=self.spacing_transform.diagonal)
            # Apply inverse
            d[key], _, new_affine = inverse_transform(
                data_array=np.asarray(d[key]),
                affine=meta_data["affine"],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False if align_corners == "none" else align_corners,
                dtype=dtype,
                output_spatial_shape=orig_size,
            )
            meta_data["affine"] = new_affine
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class Orientationd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Orientation`.

    This transform assumes the ``data`` dictionary has a key for the input
    data's metadata and contains `affine` field.  The key is formed by ``key_{meta_key_postfix}``.

    After reorienting the input array, this transform will write the new affine
    to the `affine` field of metadata which is formed by ``key_{meta_key_postfix}``.
    """

    def __init__(
        self,
        keys: KeysCollection,
        axcodes: Optional[str] = None,
        as_closest_canonical: bool = False,
        labels: Optional[Sequence[Tuple[str, str]]] = tuple(zip("LPI", "RAS")),
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = "meta_dict",
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            axcodes: N elements sequence for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical: if True, load the image as closest to canonical axis format.
            labels: optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.
            meta_keys: explicitly indicate the key of the corresponding meta data dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the meta data is a dictionary object which contains: filename, affine, original_shape, etc.
                it can be a sequence of string, map to the `keys`.
                if None, will try to construct meta_keys by `key_{meta_key_postfix}`.
            meta_key_postfix: if meta_keys is None, use `key_{postfix}` to to fetch the meta data according
                to the key data, default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
            allow_missing_keys: don't raise exception if key is missing.

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        See Also:
            `nibabel.orientations.ornt2axcodes`.

        """
        super().__init__(keys, allow_missing_keys)
        self.ornt_transform = Orientation(axcodes=axcodes, as_closest_canonical=as_closest_canonical, labels=labels)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        d: Dict = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            meta_key = meta_key or f"{key}_{meta_key_postfix}"
            # create metadata if necessary
            if meta_key not in d:
                d[meta_key] = {"affine": None}
            meta_data = d[meta_key]
            d[key], old_affine, new_affine = self.ornt_transform(d[key], affine=meta_data["affine"])
            self.push_transform(d, key, extra_info={"meta_key": meta_key, "old_affine": old_affine})
            d[meta_key]["affine"] = new_affine
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            meta_data = d[transform[InverseKeys.EXTRA_INFO]["meta_key"]]
            orig_affine = transform[InverseKeys.EXTRA_INFO]["old_affine"]
            orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
            inverse_transform = Orientation(
                axcodes=orig_axcodes,
                as_closest_canonical=False,
                labels=self.ornt_transform.labels,
            )
            # Apply inverse
            d[key], _, new_affine = inverse_transform(d[key], affine=meta_data["affine"])
            meta_data["affine"] = new_affine
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class Rotate90d(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rotate90`.
    """

    def __init__(
        self, keys: KeysCollection, k: int = 1, spatial_axes: Tuple[int, int] = (0, 1), allow_missing_keys: bool = False
    ) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
            allow_missing_keys: don't raise exception if key is missing.
        """
        super().__init__(keys, allow_missing_keys)
        self.rotator = Rotate90(k, spatial_axes)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.rotator(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Create inverse transform
            spatial_axes = self.rotator.spatial_axes
            num_times_rotated = self.rotator.k
            num_times_to_rotate = 4 - num_times_rotated
            inverse_transform = Rotate90(num_times_to_rotate, spatial_axes)
            # Might need to convert to numpy
            if isinstance(d[key], torch.Tensor):
                d[key] = torch.Tensor(d[key]).cpu().numpy()
            # Apply inverse
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandRotate90d(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate90`.
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        max_k: int = 3,
        spatial_axes: Tuple[int, int] = (0, 1),
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
            allow_missing_keys: don't raise exception if key is missing.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Optional[Any] = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        super().randomize(None)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        self.randomize()
        d = dict(data)

        rotator = Rotate90(self._rand_k, self.spatial_axes)
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = rotator(d[key])
            self.push_transform(d, key, extra_info={"rand_k": self._rand_k})
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[InverseKeys.DO_TRANSFORM]:
                # Create inverse transform
                num_times_rotated = transform[InverseKeys.EXTRA_INFO]["rand_k"]
                num_times_to_rotate = 4 - num_times_rotated
                inverse_transform = Rotate90(num_times_to_rotate, self.spatial_axes)
                # Might need to convert to numpy
                if isinstance(d[key], torch.Tensor):
                    d[key] = torch.Tensor(d[key]).cpu().numpy()
                # Apply inverse
                d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class Resized(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Resize`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        spatial_size: expected shape of spatial dimensions after resize operation.
            if the components of the `spatial_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.resizer = Resize(spatial_size=spatial_size)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, mode, align_corners in self.key_iterator(d, self.mode, self.align_corners):
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "align_corners": align_corners if align_corners is not None else "none",
                },
            )
            d[key] = self.resizer(d[key], mode=mode, align_corners=align_corners)
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            orig_size = transform[InverseKeys.ORIG_SIZE]
            mode = transform[InverseKeys.EXTRA_INFO]["mode"]
            align_corners = transform[InverseKeys.EXTRA_INFO]["align_corners"]
            # Create inverse transform
            inverse_transform = Resize(orig_size, mode, None if align_corners == "none" else align_corners)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class Affined(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Affine`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        rotate_params: Optional[Union[Sequence[float], float]] = None,
        shear_params: Optional[Union[Sequence[float], float]] = None,
        translate_params: Optional[Union[Sequence[float], float]] = None,
        scale_params: Optional[Union[Sequence[float], float]] = None,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            rotate_params: a rotation angle in radians, a scalar for 2D image, a tuple of 3 floats for 3D.
                Defaults to no rotation.
            shear_params: a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in
                pixel/voxel relative to the center of the input image. Defaults to no translation.
            scale_params: a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Defaults to no scaling.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.affine = Affine(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            spatial_size=spatial_size,
            as_tensor_output=as_tensor_output,
            device=device,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            orig_size = d[key].shape[1:]
            d[key], affine = self.affine(d[key], mode=mode, padding_mode=padding_mode)
            self.push_transform(
                d,
                key,
                orig_size=orig_size,
                extra_info={
                    "affine": affine,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                },
            )
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            orig_size = transform[InverseKeys.ORIG_SIZE]
            # Create inverse transform
            fwd_affine = transform[InverseKeys.EXTRA_INFO]["affine"]
            mode = transform[InverseKeys.EXTRA_INFO]["mode"]
            padding_mode = transform[InverseKeys.EXTRA_INFO]["padding_mode"]
            inv_affine = np.linalg.inv(fwd_affine)

            affine_grid = AffineGrid(affine=inv_affine)
            grid, _ = affine_grid(orig_size)  # type: ignore

            # Apply inverse transform
            out = self.affine.resampler(d[key], grid, mode, padding_mode)

            # Convert to numpy
            d[key] = out if isinstance(out, np.ndarray) else out.cpu().numpy()

            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandAffined(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandAffine`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        shear_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        translate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        scale_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        cache_grid: bool = False,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spatial_size: output image spatial size.
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            cache_grid: whether to cache the identity sampling grid.
                If the spatial size is not dynamically defined by input image, enabling this option could
                accelerate the transform.
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_affine = RandAffine(
            prob=1.0,  # because probability handled in this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            cache_grid=cache_grid,
            as_tensor_output=as_tensor_output,
            device=device,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandAffined":
        self.rand_affine.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.rand_affine.randomize()

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        self.randomize()

        sp_size = fall_back_tuple(self.rand_affine.spatial_size, data[self.keys[0]].shape[1:])
        # change image size or do random transform
        do_resampling = self._do_transform or (sp_size != ensure_tuple(data[self.keys[0]].shape[1:]))

        # to be consistent with the self._do_transform case (dtype and device)
        affine = torch.as_tensor(np.eye(len(sp_size) + 1), device=self.rand_affine.rand_affine_grid.device)
        grid = None
        if do_resampling:  # need to prepare grid
            grid = self.rand_affine.get_identity_grid(sp_size)
            if self._do_transform:  # add some random factors
                grid = self.rand_affine.rand_affine_grid(grid=grid)
                affine = self.rand_affine.rand_affine_grid.get_transformation_matrix()  # type: ignore[assignment]

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            self.push_transform(
                d,
                key,
                extra_info={
                    "affine": affine,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                },
            )
            # do the transform
            if do_resampling:
                d[key] = self.rand_affine.resampler(d[key], grid, mode=mode, padding_mode=padding_mode)
            # if not doing transform and and spatial size is unchanged, only need to do numpy/torch conversion
            else:
                if self.rand_affine.resampler.as_tensor_output and not isinstance(d[key], torch.Tensor):
                    d[key] = torch.Tensor(d[key])
                elif not self.rand_affine.resampler.as_tensor_output and isinstance(d[key], torch.Tensor):
                    d[key] = d[key].detach().cpu().numpy()  # type: ignore[union-attr]

        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))

        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # if transform was not performed and spatial size is None, nothing to do.
            if not transform[InverseKeys.DO_TRANSFORM] and self.rand_affine.spatial_size is None:
                out: Union[np.ndarray, torch.Tensor] = d[key]
            else:
                orig_size = transform[InverseKeys.ORIG_SIZE]
                # Create inverse transform
                fwd_affine = transform[InverseKeys.EXTRA_INFO]["affine"]
                mode = transform[InverseKeys.EXTRA_INFO]["mode"]
                padding_mode = transform[InverseKeys.EXTRA_INFO]["padding_mode"]
                inv_affine = np.linalg.inv(fwd_affine)

                affine_grid = AffineGrid(affine=inv_affine)
                grid, _ = affine_grid(orig_size)  # type: ignore

                # Apply inverse transform
                out = self.rand_affine.resampler(d[key], grid, mode, padding_mode)

            # Convert to numpy
            d[key] = out if isinstance(out, np.ndarray) else out.cpu().numpy()

            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class Rand2DElasticd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand2DElastic`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spacing: Union[Tuple[float, float], float],
        magnitude_range: Tuple[float, float],
        spatial_size: Optional[Union[Tuple[int, int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        shear_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        translate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        scale_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spacing: distance in between the control points.
            magnitude_range: 2 int numbers, the random offsets will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            spatial_size: specifying output image spatial size [h, w].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, -1)` will be adapted
                to `(32, 64)` if the second spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_2d_elastic = Rand2DElastic(
            spacing=spacing,
            magnitude_range=magnitude_range,
            prob=1.0,  # because probability controlled by this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            as_tensor_output=as_tensor_output,
            device=device,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Rand2DElasticd":
        self.rand_2d_elastic.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, spatial_size: Sequence[int]) -> None:
        super().randomize(None)
        self.rand_2d_elastic.randomize(spatial_size)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)

        sp_size = fall_back_tuple(self.rand_2d_elastic.spatial_size, data[self.keys[0]].shape[1:])
        self.randomize(spatial_size=sp_size)

        if self._do_transform:
            grid = self.rand_2d_elastic.deform_grid(spatial_size=sp_size)
            grid = self.rand_2d_elastic.rand_affine_grid(grid=grid)
            grid = torch.nn.functional.interpolate(  # type: ignore
                recompute_scale_factor=True,
                input=grid.unsqueeze(0),
                scale_factor=ensure_tuple_rep(self.rand_2d_elastic.deform_grid.spacing, 2),
                mode=InterpolateMode.BICUBIC.value,
                align_corners=False,
            )
            grid = CenterSpatialCrop(roi_size=sp_size)(grid[0])
        else:
            grid = create_grid(spatial_size=sp_size)

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.rand_2d_elastic.resampler(d[key], grid, mode=mode, padding_mode=padding_mode)
        return d


class Rand3DElasticd(RandomizableTransform, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand3DElastic`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        shear_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        translate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        scale_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            sigma_range: a Gaussian kernel with standard deviation sampled from
                ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range: the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            spatial_size: specifying output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
                if the components of the `spatial_size` are non-positive values, the transform will use the
                corresponding components of img size. For example, `spatial_size=(32, 32, -1)` will be adapted
                to `(32, 32, 64)` if the third spatial dimension size of img is `64`.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            rotate_range: angle range in radians. If element `i` is iterable, then
                `uniform[-rotate_range[i][0], rotate_range[i][1])` will be used to generate the rotation parameter
                for the ith dimension. If not, `uniform[-rotate_range[i], rotate_range[i])` will be used. This can
                be altered on a per-dimension basis. E.g., `((0,3), 1, ...)`: for dim0, rotation will be in range
                `[0, 3]`, and for dim1 `[-1, 1]` will be used. Setting a single value will use `[-x, x]` for dim0
                and nothing for the remaining dimensions.
            shear_range: shear_range with format matching `rotate_range`.
            translate_range: translate_range with format matching `rotate_range`.
            scale_range: scaling_range with format matching `rotate_range`. A value of 1.0 is added to the result.
                This allows 0 to correspond to no change (i.e., a scaling of 1).
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"reflection"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
                It also can be a sequence of string, each element corresponds to a key in ``keys``.
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device: device on which the tensor will be allocated.
            allow_missing_keys: don't raise exception if key is missing.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.rand_3d_elastic = Rand3DElastic(
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            prob=1.0,  # because probability controlled by this class
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            spatial_size=spatial_size,
            as_tensor_output=as_tensor_output,
            device=device,
        )
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "Rand3DElasticd":
        self.rand_3d_elastic.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, grid_size: Sequence[int]) -> None:
        super().randomize(None)
        self.rand_3d_elastic.randomize(grid_size)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        sp_size = fall_back_tuple(self.rand_3d_elastic.spatial_size, data[self.keys[0]].shape[1:])

        self.randomize(grid_size=sp_size)
        grid = create_grid(spatial_size=sp_size)
        if self._do_transform:
            device = self.rand_3d_elastic.device
            grid = torch.tensor(grid).to(device)
            gaussian = GaussianFilter(spatial_dims=3, sigma=self.rand_3d_elastic.sigma, truncated=3.0).to(device)
            offset = torch.tensor(self.rand_3d_elastic.rand_offset, device=device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.rand_3d_elastic.magnitude
            grid = self.rand_3d_elastic.rand_affine_grid(grid=grid)

        for key, mode, padding_mode in self.key_iterator(d, self.mode, self.padding_mode):
            d[key] = self.rand_3d_elastic.resampler(d[key], grid, mode=mode, padding_mode=padding_mode)
        return d


class Flipd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.key_iterator(d):
            self.push_transform(d, key)
            d[key] = self.flipper(d[key])
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            _ = self.get_most_recent_transform(d, key)
            # Might need to convert to numpy
            if isinstance(d[key], torch.Tensor):
                d[key] = torch.Tensor(d[key]).cpu().numpy()
            # Inverse is same as forward
            d[key] = self.flipper(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandFlipd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.spatial_axis = spatial_axis

        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        self.randomize(None)
        d = dict(data)
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = self.flipper(d[key])
            self.push_transform(d, key)
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[InverseKeys.DO_TRANSFORM]:
                # Might need to convert to numpy
                if isinstance(d[key], torch.Tensor):
                    d[key] = torch.Tensor(d[key]).cpu().numpy()
                # Inverse is same as forward
                d[key] = self.flipper(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class RandAxisFlipd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandAxisFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        allow_missing_keys: don't raise exception if key is missing.

    """

    def __init__(self, keys: KeysCollection, prob: float = 0.1, allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self._axis: Optional[int] = None

    def randomize(self, data: np.ndarray) -> None:
        super().randomize(None)
        self._axis = self.R.randint(data.ndim - 1)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        self.randomize(data=data[self.keys[0]])
        flipper = Flip(spatial_axis=self._axis)

        d = dict(data)
        for key in self.key_iterator(d):
            if self._do_transform:
                d[key] = flipper(d[key])
            self.push_transform(d, key, extra_info={"axis": self._axis})
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[InverseKeys.DO_TRANSFORM]:
                flipper = Flip(spatial_axis=transform[InverseKeys.EXTRA_INFO]["axis"])
                # Might need to convert to numpy
                if isinstance(d[key], torch.Tensor):
                    d[key] = torch.Tensor(d[key]).cpu().numpy()
                # Inverse is same as forward
                d[key] = flipper(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)
        return d


class Rotated(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rotate`.

    Args:
        keys: Keys to pick data for transformation.
        angle: Rotation angle(s) in radians.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
            It also can be a sequence of dtype or None, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        angle: Union[Sequence[float], float],
        keep_size: bool = True,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
        align_corners: Union[Sequence[bool], bool] = False,
        dtype: Union[Sequence[DtypeLike], DtypeLike] = np.float64,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.rotator = Rotate(angle=angle, keep_size=keep_size)

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            orig_size = d[key].shape[1:]
            d[key] = self.rotator(
                d[key],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
                dtype=dtype,
            )
            rot_mat = self.rotator.get_rotation_matrix()
            self.push_transform(
                d,
                key,
                orig_size=orig_size,
                extra_info={
                    "rot_mat": rot_mat,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                    "align_corners": align_corners if align_corners is not None else "none",
                },
            )
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key, dtype in self.key_iterator(d, self.dtype):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            fwd_rot_mat = transform[InverseKeys.EXTRA_INFO]["rot_mat"]
            mode = transform[InverseKeys.EXTRA_INFO]["mode"]
            padding_mode = transform[InverseKeys.EXTRA_INFO]["padding_mode"]
            align_corners = transform[InverseKeys.EXTRA_INFO]["align_corners"]
            inv_rot_mat = np.linalg.inv(fwd_rot_mat)

            xform = AffineTransform(
                normalized=False,
                mode=mode,
                padding_mode=padding_mode,
                align_corners=False if align_corners == "none" else align_corners,
                reverse_indexing=True,
            )
            output = xform(
                torch.as_tensor(np.ascontiguousarray(d[key]).astype(dtype)).unsqueeze(0),
                torch.as_tensor(np.ascontiguousarray(inv_rot_mat).astype(dtype)),
                spatial_size=transform[InverseKeys.ORIG_SIZE],
            )
            d[key] = np.asarray(output.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandRotated(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandRotate`
    Randomly rotates the input arrays.

    Args:
        keys: Keys to pick data for transformation.
        range_x: Range of rotation angle in radians in the plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y: Range of rotation angle in radians in the plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y).
        range_z: Range of rotation angle in radians in the plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z).
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            It also can be a sequence of bool, each element corresponds to a key in ``keys``.
        dtype: data type for resampling computation. Defaults to ``np.float64`` for best precision.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.
            It also can be a sequence of dtype or None, each element corresponds to a key in ``keys``.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        range_x: Union[Tuple[float, float], float] = 0.0,
        range_y: Union[Tuple[float, float], float] = 0.0,
        range_z: Union[Tuple[float, float], float] = 0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.BORDER,
        align_corners: Union[Sequence[bool], bool] = False,
        dtype: Union[Sequence[DtypeLike], DtypeLike] = np.float64,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.range_x = ensure_tuple(range_x)
        if len(self.range_x) == 1:
            self.range_x = tuple(sorted([-self.range_x[0], self.range_x[0]]))
        self.range_y = ensure_tuple(range_y)
        if len(self.range_y) == 1:
            self.range_y = tuple(sorted([-self.range_y[0], self.range_y[0]]))
        self.range_z = ensure_tuple(range_z)
        if len(self.range_z) == 1:
            self.range_z = tuple(sorted([-self.range_z[0], self.range_z[0]]))

        self.keep_size = keep_size
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        self.randomize()
        d = dict(data)
        angle: Union[Sequence[float], float] = self.x if d[self.keys[0]].ndim == 3 else (self.x, self.y, self.z)
        rotator = Rotate(
            angle=angle,
            keep_size=self.keep_size,
        )
        for key, mode, padding_mode, align_corners, dtype in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners, self.dtype
        ):
            orig_size = d[key].shape[1:]
            if self._do_transform:
                d[key] = rotator(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                    dtype=dtype,
                )
                rot_mat = rotator.get_rotation_matrix()
            else:
                rot_mat = np.eye(d[key].ndim)
            self.push_transform(
                d,
                key,
                orig_size=orig_size,
                extra_info={
                    "rot_mat": rot_mat,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                    "align_corners": align_corners if align_corners is not None else "none",
                },
            )
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key, dtype in self.key_iterator(d, self.dtype):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[InverseKeys.DO_TRANSFORM]:
                # Create inverse transform
                fwd_rot_mat = transform[InverseKeys.EXTRA_INFO]["rot_mat"]
                mode = transform[InverseKeys.EXTRA_INFO]["mode"]
                padding_mode = transform[InverseKeys.EXTRA_INFO]["padding_mode"]
                align_corners = transform[InverseKeys.EXTRA_INFO]["align_corners"]
                inv_rot_mat = np.linalg.inv(fwd_rot_mat)

                xform = AffineTransform(
                    normalized=False,
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=False if align_corners == "none" else align_corners,
                    reverse_indexing=True,
                )
                output = xform(
                    torch.as_tensor(np.ascontiguousarray(d[key]).astype(dtype)).unsqueeze(0),
                    torch.as_tensor(np.ascontiguousarray(inv_rot_mat).astype(dtype)),
                    spatial_size=transform[InverseKeys.ORIG_SIZE],
                )
                d[key] = np.asarray(output.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class Zoomd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Zoom`.

    Args:
        keys: Keys to pick data for transformation.
        zoom: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"constant"``, ``"edge``", ``"linear_ramp``", ``"maximum``", ``"mean``", `"median``",
            ``"minimum``", `"reflect``", ``"symmetric``", ``"wrap``", ``"empty``", ``"<function>``"}
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        zoom: Union[Sequence[float], float],
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        padding_mode: NumpyPadModeSequence = NumpyPadMode.EDGE,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.zoomer = Zoom(zoom=zoom, keep_size=keep_size)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key, mode, padding_mode, align_corners in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners
        ):
            self.push_transform(
                d,
                key,
                extra_info={
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                    "align_corners": align_corners if align_corners is not None else "none",
                },
            )
            d[key] = self.zoomer(
                d[key],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            zoom = np.array(self.zoomer.zoom)
            inverse_transform = Zoom(zoom=1 / zoom, keep_size=self.zoomer.keep_size)
            mode = transform[InverseKeys.EXTRA_INFO]["mode"]
            padding_mode = transform[InverseKeys.EXTRA_INFO]["padding_mode"]
            align_corners = transform[InverseKeys.EXTRA_INFO]["align_corners"]
            # Apply inverse
            d[key] = inverse_transform(
                d[key],
                mode=mode,
                padding_mode=padding_mode,
                align_corners=None if align_corners == "none" else align_corners,
            )
            # Size might be out by 1 voxel so pad
            d[key] = SpatialPad(transform[InverseKeys.ORIG_SIZE], mode="edge")(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class RandZoomd(RandomizableTransform, MapTransform, InvertibleTransform):
    """
    Dict-based version :py:class:`monai.transforms.RandZoom`.

    Args:
        keys: Keys to pick data for transformation.
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
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            It also can be a sequence of string, each element corresponds to a key in ``keys``.
        padding_mode: {``"constant"``, ``"edge``", ``"linear_ramp``", ``"maximum``", ``"mean``", `"median``",
            ``"minimum``", `"reflect``", ``"symmetric``", ``"wrap``", ``"empty``", ``"<function>``"}
            The mode to pad data after zooming.
            See also: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
            It also can be a sequence of bool or None, each element corresponds to a key in ``keys``.
        keep_size: Should keep original size (pad if needed), default is True.
        allow_missing_keys: don't raise exception if key is missing.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        min_zoom: Union[Sequence[float], float] = 0.9,
        max_zoom: Union[Sequence[float], float] = 1.1,
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        padding_mode: NumpyPadModeSequence = NumpyPadMode.EDGE,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        keep_size: bool = True,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)
        self.min_zoom = ensure_tuple(min_zoom)
        self.max_zoom = ensure_tuple(max_zoom)
        if len(self.min_zoom) != len(self.max_zoom):
            raise AssertionError("min_zoom and max_zoom must have same length.")

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.keep_size = keep_size

        self._zoom: Sequence[float] = [1.0]

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self._zoom = [self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom)]

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        # match the spatial dim of first item
        self.randomize()
        d = dict(data)

        img_dims = data[self.keys[0]].ndim
        if len(self._zoom) == 1:
            # to keep the spatial shape ratio, use same random zoom factor for all dims
            self._zoom = ensure_tuple_rep(self._zoom[0], img_dims - 1)
        elif len(self._zoom) == 2 and img_dims > 3:
            # if 2 zoom factors provided for 3D data, use the first factor for H and W dims, second factor for D dim
            self._zoom = ensure_tuple_rep(self._zoom[0], img_dims - 2) + ensure_tuple(self._zoom[-1])
        zoomer = Zoom(self._zoom, keep_size=self.keep_size)
        for key, mode, padding_mode, align_corners in self.key_iterator(
            d, self.mode, self.padding_mode, self.align_corners
        ):
            self.push_transform(
                d,
                key,
                extra_info={
                    "zoom": self._zoom,
                    "mode": mode.value if isinstance(mode, Enum) else mode,
                    "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                    "align_corners": align_corners if align_corners is not None else "none",
                },
            )
            if self._do_transform:
                d[key] = zoomer(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=align_corners,
                )
        return d

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.key_iterator(d):
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform[InverseKeys.DO_TRANSFORM]:
                # Create inverse transform
                zoom = np.array(transform[InverseKeys.EXTRA_INFO]["zoom"])
                mode = transform[InverseKeys.EXTRA_INFO]["mode"]
                padding_mode = transform[InverseKeys.EXTRA_INFO]["padding_mode"]
                align_corners = transform[InverseKeys.EXTRA_INFO]["align_corners"]
                inverse_transform = Zoom(zoom=1 / zoom, keep_size=self.keep_size)
                # Apply inverse
                d[key] = inverse_transform(
                    d[key],
                    mode=mode,
                    padding_mode=padding_mode,
                    align_corners=None if align_corners == "none" else align_corners,
                )
                # Size might be out by 1 voxel so pad
                d[key] = SpatialPad(transform[InverseKeys.ORIG_SIZE], mode="edge")(d[key])
            # Remove the applied transform
            self.pop_transform(d, key)

        return d


class AddCoordinateChannelsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddCoordinateChannels`.
    """

    def __init__(self, keys: KeysCollection, spatial_channels: Sequence[int], allow_missing_keys: bool = False) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            allow_missing_keys: don't raise exception if key is missing.
            spatial_channels: the spatial dimensions that are to have their coordinates encoded in a channel and
                appended to the input. E.g., `(1,2,3)` will append three channels to the input, encoding the
                coordinates of the input's three spatial dimensions. It is assumed dimension 0 is the channel.

        """
        super().__init__(keys, allow_missing_keys)
        self.add_coordinate_channels = AddCoordinateChannels(spatial_channels)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.add_coordinate_channels(d[key])
        return d


SpacingD = SpacingDict = Spacingd
OrientationD = OrientationDict = Orientationd
Rotate90D = Rotate90Dict = Rotate90d
RandRotate90D = RandRotate90Dict = RandRotate90d
ResizeD = ResizeDict = Resized
AffineD = AffineDict = Affined
RandAffineD = RandAffineDict = RandAffined
Rand2DElasticD = Rand2DElasticDict = Rand2DElasticd
Rand3DElasticD = Rand3DElasticDict = Rand3DElasticd
FlipD = FlipDict = Flipd
RandFlipD = RandFlipDict = RandFlipd
RandAxisFlipD = RandAxisFlipDict = RandAxisFlipd
RotateD = RotateDict = Rotated
RandRotateD = RandRotateDict = RandRotated
ZoomD = ZoomDict = Zoomd
RandZoomD = RandZoomDict = RandZoomd
AddCoordinateChannelsD = AddCoordinateChannelsDict = AddCoordinateChannelsd
