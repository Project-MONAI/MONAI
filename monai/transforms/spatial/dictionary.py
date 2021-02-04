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
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from monai.config import DtypeLike, KeysCollection
from monai.networks.layers import AffineTransform
from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.croppad.array import CenterSpatialCrop, SpatialPad
from monai.transforms.spatial.array import (
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
    AffineGrid,
)
from monai.transforms.transform import InvertibleTransform, MapTransform, Randomizable
from monai.transforms.utils import create_grid
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    optional_import,
)

nib, _ = optional_import("nibabel")

__all__ = [
    "Spacingd",
    "Orientationd",
    "Rotate90d",
    "RandRotate90d",
    "Resized",
    "RandAffined",
    "Rand2DElasticd",
    "Rand3DElasticd",
    "Flipd",
    "RandFlipd",
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
    "RotateD",
    "RotateDict",
    "RandRotateD",
    "RandRotateDict",
    "ZoomD",
    "ZoomDict",
    "RandZoomD",
    "RandZoomDict",
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
        meta_key_postfix: str = "meta_dict",
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
            meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        """
        super().__init__(keys)
        self.spacing_transform = Spacing(pixdim, diagonal=diagonal)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        d: Dict = dict(data)
        for idx, key in enumerate(self.keys):
            meta_data_key = f"{key}_{self.meta_key_postfix}"
            meta_data = d[meta_data_key]
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            d[key], old_affine, new_affine = self.spacing_transform(
                data_array=np.asarray(d[key]),
                affine=meta_data["affine"],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                align_corners=self.align_corners[idx],
                dtype=self.dtype[idx],
            )
            self.append_applied_transforms(d, key, extra_info={"meta_data_key": meta_data_key, "old_affine": old_affine})
            # set the 'affine' key
            meta_data["affine"] = new_affine
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "pixdim": self.spacing_transform.pixdim,
            "diagonal": self.spacing_transform.diagonal,
            "mode": self.mode[idx],
            "padding_mode": self.padding_mode[idx],
            "align_corners": self.align_corners[idx],
            "dtype": self.dtype[idx],
            "meta_key_postfix": self.meta_key_postfix
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            init_args = transform["init_args"]
            if init_args["diagonal"]:
                raise RuntimeError(
                    "Spacingd:inverse not yet implemented for diagonal=True. " +
                    "Please raise a github issue if you need this feature")
            # Create inverse transform
            meta_data = d[transform["extra_info"]["meta_data_key"]]
            orig_pixdim = np.sqrt(np.sum(np.square(transform["extra_info"]["old_affine"]), 0))[:-1]
            inverse_transform = Spacing(orig_pixdim, diagonal=init_args["diagonal"])
            # Apply inverse
            d[key], _, new_affine = inverse_transform(
                data_array=np.asarray(d[key]),
                affine=meta_data["affine"],
                mode=init_args["mode"],
                padding_mode=init_args["padding_mode"],
                align_corners=init_args["align_corners"],
                dtype=init_args["dtype"],
            )
            meta_data["affine"] = new_affine
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

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
        meta_key_postfix: str = "meta_dict",
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
            meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.

        Raises:
            TypeError: When ``meta_key_postfix`` is not a ``str``.

        See Also:
            `nibabel.orientations.ornt2axcodes`.

        """
        super().__init__(keys)
        self.ornt_transform = Orientation(axcodes=axcodes, as_closest_canonical=as_closest_canonical, labels=labels)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_key_postfix = meta_key_postfix

    def __call__(
        self, data: Mapping[Union[Hashable, str], Dict[str, np.ndarray]]
    ) -> Dict[Union[Hashable, str], Union[np.ndarray, Dict[str, np.ndarray]]]:
        d: Dict = dict(data)
        for key in self.keys:
            meta_data_key = f"{key}_{self.meta_key_postfix}"
            meta_data = d[meta_data_key]
            d[key], old_affine, new_affine = self.ornt_transform(d[key], affine=meta_data["affine"])
            self.append_applied_transforms(d, key, extra_info={"meta_data_key": meta_data_key, "old_affine": old_affine})
            d[meta_data_key]["affine"] = new_affine
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "axcodes": self.ornt_transform.axcodes,
            "as_closest_canonical": self.ornt_transform.as_closest_canonical,
            "labels": self.ornt_transform.labels,
            "meta_key_postfix": self.meta_key_postfix,
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            meta_data = d[transform["extra_info"]["meta_data_key"]]
            orig_affine = transform["extra_info"]["old_affine"]
            orig_axcodes = nib.orientations.aff2axcodes(orig_affine)
            inverse_transform = Orientation(
                axcodes=orig_axcodes,
                as_closest_canonical=transform["init_args"]["as_closest_canonical"],
                labels=transform["init_args"]["labels"],
            )
            # Apply inverse
            d[key], _, new_affine = inverse_transform(d[key], affine=meta_data["affine"])
            meta_data["affine"] = new_affine
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

        return d


class Rotate90d(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rotate90`.
    """

    def __init__(self, keys: KeysCollection, k: int = 1, spatial_axes: Tuple[int, int] = (0, 1)) -> None:
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        super().__init__(keys)
        self.rotator = Rotate90(k, spatial_axes)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            self.append_applied_transforms(d, key)
            d[key] = self.rotator(d[key])
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "k": self.rotator.k,
            "spatial_axes": self.rotator.spatial_axes,
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            spatial_axes = transform["init_args"]["spatial_axes"]
            num_times_rotated = transform["init_args"]["k"]
            num_times_to_rotate = 4 - num_times_rotated
            inverse_transform = Rotate90(num_times_to_rotate, spatial_axes)
            # Apply inverse
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

        return d


class RandRotate90d(Randomizable, MapTransform, InvertibleTransform):
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
        """
        MapTransform.__init__(self, keys)
        Randomizable.__init__(self, min(max(prob, 0.0), 1.0))

        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._rand_k = 0

    def randomize(self, data: Optional[Any] = None) -> None:
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:
        self.randomize()
        if not self._do_transform:
            for key in self.keys:
                self.append_applied_transforms(data, key)
            return data

        rotator = Rotate90(self._rand_k, self.spatial_axes)
        d = dict(data)
        for key in self.keys:
            d[key] = rotator(d[key])
            self.append_applied_transforms(d, key, extra_info={"rand_k": self._rand_k})
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "prob": self.prob,
            "max_k": self.max_k,
            "spatial_axes": self.spatial_axes,
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform["do_transform"]:
                # Create inverse transform
                spatial_axes = transform["init_args"]["spatial_axes"]
                num_times_rotated = transform["extra_info"]["rand_k"]
                num_times_to_rotate = 4 - num_times_rotated
                inverse_transform = Rotate90(num_times_to_rotate, spatial_axes)
                # Apply inverse
                d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

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
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Union[Sequence[int], int],
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
    ) -> None:
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.resizer = Resize(spatial_size=spatial_size)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            self.append_applied_transforms(d, key)
            d[key] = self.resizer(d[key], mode=self.mode[idx], align_corners=self.align_corners[idx])
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "spatial_size": self.resizer.spatial_size,
            "mode": self.mode[idx],
            "align_corners": self.align_corners[idx],
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            orig_size = transform["orig_size"]
            mode = transform["init_args"]["mode"]
            align_corners = transform["init_args"]["align_corners"]
            # Create inverse transform
            inverse_transform = Resize(orig_size, mode, align_corners)
            # Apply inverse transform
            d[key] = inverse_transform(d[key])
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

        return d


class RandAffined(Randomizable, MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandAffine`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[float], float]] = None,
        shear_range: Optional[Union[Sequence[float], float]] = None,
        translate_range: Optional[Union[Sequence[float], float]] = None,
        scale_range: Optional[Union[Sequence[float], float]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
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
            scale_range: scaling_range with format matching `rotate_range`.
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

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
        """
        MapTransform.__init__(self, keys)
        Randomizable.__init__(self, prob)
        self.rand_affine = RandAffine(
            prob=prob,
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
    ) -> "RandAffined":
        self.rand_affine.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, data: Optional[Any] = None) -> None:
        self.rand_affine.randomize()
        self.prob = self.rand_affine.prob
        self._do_transform = self.rand_affine._do_transform

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        self.randomize()

        sp_size = fall_back_tuple(self.rand_affine.spatial_size, data[self.keys[0]].shape[1:])
        if self.rand_affine._do_transform:
            grid, affine = self.rand_affine.rand_affine_grid(spatial_size=sp_size)
        else:
            grid = create_grid(spatial_size=sp_size)
            affine = np.eye(len(sp_size) + 1)

        for idx, key in enumerate(self.keys):
            self.append_applied_transforms(d, key, idx, extra_info={"affine": affine, "orig_was_numpy": isinstance(d[key], np.ndarray)})
            d[key] = self.rand_affine.resampler(d[key], grid, mode=self.mode[idx], padding_mode=self.padding_mode[idx])
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "spatial_size": self.rand_affine.spatial_size,
            "prob": self.rand_affine.prob,
            "rotate_range": self.rand_affine.rand_affine_grid.rotate_range,
            "shear_range": self.rand_affine.rand_affine_grid.shear_range,
            "translate_range": self.rand_affine.rand_affine_grid.translate_range,
            "scale_range": self.rand_affine.rand_affine_grid.scale_range,
            "mode": self.rand_affine.mode,
            "padding_mode": self.rand_affine.padding_mode,
            "as_tensor_output": self.rand_affine.resampler.as_tensor_output,
            "device": self.rand_affine.resampler.device,
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))

        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            extra_info = transform["extra_info"]
            init_args = transform["init_args"]
            orig_size = transform["orig_size"]
            # Create inverse transform
            fwd_affine = extra_info["affine"]
            inv_affine = np.linalg.inv(fwd_affine)

            affine_grid = AffineGrid(affine=inv_affine)
            grid = affine_grid(orig_size)

            # Apply inverse transform
            d[key] = self.rand_affine.resampler(d[key], grid, init_args["mode"], init_args["padding_mode"])
            if extra_info["orig_was_numpy"]:
                d[key] = d[key].cpu().numpy()
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

        return d


class Rand2DElasticd(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand2DElastic`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spacing: Union[Tuple[float, float], float],
        magnitude_range: Tuple[float, float],
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[float], float]] = None,
        shear_range: Optional[Union[Sequence[float], float]] = None,
        translate_range: Optional[Union[Sequence[float], float]] = None,
        scale_range: Optional[Union[Sequence[float], float]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
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
            scale_range: scaling_range with format matching `rotate_range`.
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

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        super().__init__(keys)
        self.rand_2d_elastic = Rand2DElastic(
            spacing=spacing,
            magnitude_range=magnitude_range,
            prob=prob,
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
        self.rand_2d_elastic.randomize(spatial_size)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)

        sp_size = fall_back_tuple(self.rand_2d_elastic.spatial_size, data[self.keys[0]].shape[1:])
        self.randomize(spatial_size=sp_size)

        if self.rand_2d_elastic._do_transform:
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

        for idx, key in enumerate(self.keys):
            d[key] = self.rand_2d_elastic.resampler(
                d[key], grid, mode=self.mode[idx], padding_mode=self.padding_mode[idx]
            )
        return d


class Rand3DElasticd(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand3DElastic`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        sigma_range: Tuple[float, float],
        magnitude_range: Tuple[float, float],
        spatial_size: Optional[Union[Sequence[int], int]] = None,
        prob: float = 0.1,
        rotate_range: Optional[Union[Sequence[float], float]] = None,
        shear_range: Optional[Union[Sequence[float], float]] = None,
        translate_range: Optional[Union[Sequence[float], float]] = None,
        scale_range: Optional[Union[Sequence[float], float]] = None,
        mode: GridSampleModeSequence = GridSampleMode.BILINEAR,
        padding_mode: GridSamplePadModeSequence = GridSamplePadMode.REFLECTION,
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
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
            scale_range: scaling_range with format matching `rotate_range`.
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

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        super().__init__(keys)
        self.rand_3d_elastic = Rand3DElastic(
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            prob=prob,
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
        self.rand_3d_elastic.randomize(grid_size)

    def __call__(
        self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[Hashable, Union[np.ndarray, torch.Tensor]]:
        d = dict(data)
        sp_size = fall_back_tuple(self.rand_3d_elastic.spatial_size, data[self.keys[0]].shape[1:])

        self.randomize(grid_size=sp_size)
        grid = create_grid(spatial_size=sp_size)
        if self.rand_3d_elastic._do_transform:
            device = self.rand_3d_elastic.device
            grid = torch.tensor(grid).to(device)
            gaussian = GaussianFilter(spatial_dims=3, sigma=self.rand_3d_elastic.sigma, truncated=3.0).to(device)
            offset = torch.tensor(self.rand_3d_elastic.rand_offset, device=device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.rand_3d_elastic.magnitude
            grid = self.rand_3d_elastic.rand_affine_grid(grid=grid)

        for idx, key in enumerate(self.keys):
            d[key] = self.rand_3d_elastic.resampler(
                d[key], grid, mode=self.mode[idx], padding_mode=self.padding_mode[idx]
            )
        return d


class Flipd(MapTransform, InvertibleTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, keys: KeysCollection, spatial_axis: Optional[Union[Sequence[int], int]] = None) -> None:
        super().__init__(keys)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for key in self.keys:
            self.append_applied_transforms(d, key)
            d[key] = self.flipper(d[key])
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "spatial_axis": self.flipper.spatial_axis,
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            _ = self.get_most_recent_transform(d, key)
            # Inverse is same as forward
            d[key] = self.flipper(d[key])
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

        return d


class RandFlipd(Randomizable, MapTransform, InvertibleTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of flipping.
        spatial_axis: Spatial axes along which to flip over. Default is None.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        spatial_axis: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        MapTransform.__init__(self, keys)
        Randomizable.__init__(self, prob)
        self.spatial_axis = spatial_axis

        self.flipper = Flip(spatial_axis=spatial_axis)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._do_transform = self.R.random_sample() < self.prob

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        self.randomize()
        d = dict(data)
        for key in self.keys:
            if self._do_transform:
                d[key] = self.flipper(d[key])
            self.append_applied_transforms(d, key)

        return d


    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "spatial_axis": self.flipper.spatial_axis,
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform["do_transform"]:
                # Inverse is same as forward
                d[key] = self.flipper(d[key])
                # Remove the applied transform
                self.remove_most_recent_transform(d, key)

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
    ) -> None:
        super().__init__(keys)
        self.rotator = Rotate(angle=angle, keep_size=keep_size)

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            orig_size = d[key].shape[1:]
            d[key], rot_mat = self.rotator(
                d[key],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                align_corners=self.align_corners[idx],
                dtype=self.dtype[idx],
                return_rotation_matrix=True,
            )
            self.append_applied_transforms(d, key, idx, orig_size=orig_size, extra_info={"rot_mat": rot_mat})
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "angle": self.rotator.angle,
            "keep_size": self.rotator.keep_size,
            "mode": self.mode[idx],
            "padding_mode": self.padding_mode[idx],
            "align_corners": self.align_corners[idx],
            "dtype": self.dtype[idx],
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            init_args = transform["init_args"]
            # Create inverse transform
            fwd_rot_mat = transform["extra_info"]["rot_mat"]
            inv_rot_mat = np.linalg.inv(fwd_rot_mat)

            xform = AffineTransform(
                normalized=False,
                mode=init_args["mode"],
                padding_mode=init_args["padding_mode"],
                align_corners=init_args["align_corners"],
                reverse_indexing=True,
            )
            dtype = init_args["dtype"]
            output = xform(
                torch.as_tensor(np.ascontiguousarray(d[key]).astype(dtype)).unsqueeze(0),
                torch.as_tensor(np.ascontiguousarray(inv_rot_mat).astype(dtype)),
                spatial_size=transform["orig_size"],
            )
            d[key] = np.asarray(output.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

        return d


class RandRotated(Randomizable, MapTransform, InvertibleTransform):
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
    ) -> None:
        MapTransform.__init__(self, keys)
        Randomizable.__init__(self, prob)
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
        self._do_transform = self.R.random_sample() < self.prob
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            for key in self.keys:
                self.append_applied_transforms(d, key)
            return d
        angle: Sequence = self.x if d[self.keys[0]].ndim == 3 else (self.x, self.y, self.z)
        rotator = Rotate(
            angle=angle,
            keep_size=self.keep_size,
        )
        for idx, key in enumerate(self.keys):
            orig_size = d[key].shape[1:]
            d[key], rot_mat = rotator(
                d[key],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                align_corners=self.align_corners[idx],
                dtype=self.dtype[idx],
                return_rotation_matrix=True,
            )
            self.append_applied_transforms(d, key, idx, orig_size=orig_size, extra_info={"rot_mat": rot_mat})
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "range_x": self.range_x,
            "range_y": self.range_y,
            "range_z": self.range_z,
            "prob": self.prob,
            "keep_size": self.keep_size,
            "mode": self.mode[idx],
            "padding_mode": self.padding_mode[idx],
            "align_corners": self.align_corners[idx],
            "dtype": self.dtype[idx],
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            # Check if random transform was actually performed (based on `prob`)
            if transform["do_transform"]:
                init_args = transform["init_args"]
                # Create inverse transform
                fwd_rot_mat = transform["extra_info"]["rot_mat"]
                inv_rot_mat = np.linalg.inv(fwd_rot_mat)

                xform = AffineTransform(
                    normalized=False,
                    mode=init_args["mode"],
                    padding_mode=init_args["padding_mode"],
                    align_corners=init_args["align_corners"],
                    reverse_indexing=True,
                )
                dtype = init_args["dtype"]
                output = xform(
                    torch.as_tensor(np.ascontiguousarray(d[key]).astype(dtype)).unsqueeze(0),
                    torch.as_tensor(np.ascontiguousarray(inv_rot_mat).astype(dtype)),
                    spatial_size=transform["orig_size"],
                )
                d[key] = np.asarray(output.squeeze(0).detach().cpu().numpy(), dtype=np.float32)
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

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
    """

    def __init__(
        self,
        keys: KeysCollection,
        zoom: Union[Sequence[float], float],
        mode: InterpolateModeSequence = InterpolateMode.AREA,
        padding_mode: NumpyPadModeSequence = NumpyPadMode.EDGE,
        align_corners: Union[Sequence[Optional[bool]], Optional[bool]] = None,
        keep_size: bool = True,
    ) -> None:
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = ensure_tuple_rep(align_corners, len(self.keys))
        self.zoomer = Zoom(zoom=zoom, keep_size=keep_size)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        for idx, key in enumerate(self.keys):
            self.append_applied_transforms(d, key)
            d[key] = self.zoomer(
                d[key],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                align_corners=self.align_corners[idx],
            )
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "zoom": self.zoomer.zoom,
            "mode": self.mode[idx],
            "padding_mode": self.padding_mode[idx],
            "align_corners": self.align_corners[idx],
            "keep_size": self.zoomer.keep_size,
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            init_args = transform["init_args"]
            zoom = np.array(init_args["zoom"])
            inverse_transform = Zoom(zoom=1 / zoom, keep_size=init_args["keep_size"])
            # Apply inverse
            d[key] = inverse_transform(
                d[key],
                mode=init_args["mode"],
                padding_mode=init_args["padding_mode"],
                align_corners=init_args["align_corners"],
            )
            # Size might be out by 1 voxel so pad
            d[key] = SpatialPad(transform["orig_size"])(d[key])
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

        return d

class RandZoomd(Randomizable, MapTransform, InvertibleTransform):
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
    ) -> None:
        MapTransform.__init__(self, keys)
        Randomizable.__init__(self, prob)
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
        self._do_transform = self.R.random_sample() < self.prob
        self._zoom = [self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom)]

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        # match the spatial dim of first item
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            for key in self.keys:
                self.append_applied_transforms(d, key)
            return d

        img_dims = data[self.keys[0]].ndim
        if len(self._zoom) == 1:
            # to keep the spatial shape ratio, use same random zoom factor for all dims
            self._zoom = ensure_tuple_rep(self._zoom[0], img_dims - 1)
        elif len(self._zoom) == 2 and img_dims > 3:
            # if 2 zoom factors provided for 3D data, use the first factor for H and W dims, second factor for D dim
            self._zoom = ensure_tuple_rep(self._zoom[0], img_dims - 2) + ensure_tuple(self._zoom[-1])
        zoomer = Zoom(self._zoom, keep_size=self.keep_size)
        for idx, key in enumerate(self.keys):
            self.append_applied_transforms(d, key, extra_info={"zoom": self._zoom})
            d[key] = zoomer(
                d[key],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                align_corners=self.align_corners[idx],
            )
        return d

    def get_input_args(self, key: Hashable, idx: int = 0) -> dict:
        return {
            "keys": key,
            "prob": self.prob,
            "min_zoom": self.min_zoom,
            "max_zoom": self.max_zoom,
            "mode": self.mode[idx],
            "padding_mode": self.padding_mode[idx],
            "align_corners": self.align_corners[idx],
            "keep_size": self.keep_size,
        }

    def inverse(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = deepcopy(dict(data))
        for key in self.keys:
            transform = self.get_most_recent_transform(d, key)
            # Create inverse transform
            init_args = transform["init_args"]
            zoom = np.array(transform["extra_info"]["zoom"])
            inverse_transform = Zoom(zoom=1 / zoom, keep_size=init_args["keep_size"])
            # Apply inverse
            d[key] = inverse_transform(
                d[key],
                mode=init_args["mode"],
                padding_mode=init_args["padding_mode"],
                align_corners=init_args["align_corners"],
            )
            # Size might be out by 1 voxel so pad
            d[key] = SpatialPad(transform["orig_size"])(d[key])
            # Remove the applied transform
            self.remove_most_recent_transform(d, key)

        return d


SpacingD = SpacingDict = Spacingd
OrientationD = OrientationDict = Orientationd
Rotate90D = Rotate90Dict = Rotate90d
RandRotate90D = RandRotate90Dict = RandRotate90d
ResizeD = ResizeDict = Resized
RandAffineD = RandAffineDict = RandAffined
Rand2DElasticD = Rand2DElasticDict = Rand2DElasticd
Rand3DElasticD = Rand3DElasticDict = Rand3DElasticd
FlipD = FlipDict = Flipd
RandFlipD = RandFlipDict = RandFlipd
RotateD = RotateDict = Rotated
RandRotateD = RandRotateDict = RandRotated
ZoomD = ZoomDict = Zoomd
RandZoomD = RandZoomDict = RandZoomd
