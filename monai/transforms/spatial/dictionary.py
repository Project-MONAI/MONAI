# Copyright 2020 MONAI Consortium
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

from typing import Optional, Sequence, Union

import numpy as np
import torch

from monai.config.type_definitions import KeysCollection

from monai.networks.layers.simplelayers import GaussianFilter
from monai.transforms.compose import MapTransform, Randomizable
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
    _torch_interp,
)
from monai.transforms.utils import create_grid
from monai.utils.misc import ensure_tuple, ensure_tuple_rep


class Spacingd(MapTransform):
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
        pixdim,
        diagonal: bool = False,
        mode: Union[Sequence[str], str] = "bilinear",
        padding_mode: Union[Sequence[str], str] = "border",
        dtype: Optional[np.dtype] = None,
        meta_key_postfix: str = "meta_dict",
    ):
        """
        Args:
            pixdim (sequence of floats): output voxel spacing.
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
                For a sequence each element corresponds to a key in ``keys``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                For a sequence each element corresponds to a key in ``keys``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype (None or np.dtype or sequence of np.dtype): output array data type.
                Defaults to None to use input data's dtype.
            meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.
        """
        super().__init__(keys)
        self.spacing_transform = Spacing(pixdim, diagonal=diagonal)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.dtype = ensure_tuple_rep(dtype, len(self.keys))
        if not isinstance(meta_key_postfix, str):
            raise ValueError("meta_key_postfix must be a string.")
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            meta_data = d[f"{key}_{self.meta_key_postfix}"]
            # resample array of each corresponding key
            # using affine fetched from d[affine_key]
            d[key], _, new_affine = self.spacing_transform(
                data_array=d[key],
                affine=meta_data["affine"],
                mode=self.mode[idx],
                padding_mode=self.padding_mode[idx],
                dtype=self.dtype[idx],
            )
            # set the 'affine' key
            meta_data["affine"] = new_affine
        return d


class Orientationd(MapTransform):
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
        axcodes=None,
        as_closest_canonical: bool = False,
        labels=tuple(zip("LPI", "RAS")),
        meta_key_postfix: str = "meta_dict",
    ):
        """
        Args:
            axcodes (N elements sequence): for spatial ND input's orientation.
                e.g. axcodes='RAS' represents 3D orientation:
                (Left, Right), (Posterior, Anterior), (Inferior, Superior).
                default orientation labels options are: 'L' and 'R' for the first dimension,
                'P' and 'A' for the second, 'I' and 'S' for the third.
            as_closest_canonical (boo): if True, load the image as closest to canonical axis format.
            labels : optional, None or sequence of (2,) sequences
                (2,) sequences are labels for (beginning, end) of output axis.
                Defaults to ``(('L', 'R'), ('P', 'A'), ('I', 'S'))``.
            meta_key_postfix: use `key_{postfix}` to to fetch the meta data according to the key data,
                default is `meta_dict`, the meta data is a dictionary object.
                For example, to handle key `image`,  read/write affine matrices from the
                metadata `image_meta_dict` dictionary's `affine` field.

        See Also:
            `nibabel.orientations.ornt2axcodes`.
        """
        super().__init__(keys)
        self.ornt_transform = Orientation(axcodes=axcodes, as_closest_canonical=as_closest_canonical, labels=labels)
        if not isinstance(meta_key_postfix, str):
            raise ValueError("meta_key_postfix must be a string.")
        self.meta_key_postfix = meta_key_postfix

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            meta_data = d[f"{key}_{self.meta_key_postfix}"]
            d[key], _, new_affine = self.ornt_transform(d[key], affine=meta_data["affine"])
            meta_data["affine"] = new_affine
        return d


class Rotate90d(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rotate90`.
    """

    def __init__(self, keys: KeysCollection, k: int = 1, spatial_axes=(0, 1)):
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes (2 ints): defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        super().__init__(keys)
        self.rotator = Rotate90(k, spatial_axes)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.rotator(d[key])
        return d


class RandRotate90d(Randomizable, MapTransform):
    """Dictionary-based version :py:class:`monai.transforms.RandRotate90`.
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    def __init__(self, keys: KeysCollection, prob: float = 0.1, max_k: int = 3, spatial_axes=(0, 1)):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array.)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes (2 ints): defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        super().__init__(keys)

        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, data):
        self.randomize()
        if not self._do_transform:
            return data

        rotator = Rotate90(self._rand_k, self.spatial_axes)
        d = dict(data)
        for key in self.keys:
            d[key] = rotator(d[key])
        return d


class Resized(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Resize`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        spatial_size (tuple or list): expected shape of spatial dimensions after resize operation.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            For a sequence each element corresponds to a key in ``keys``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        align_corners (optional bool): This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size,
        mode: Union[Sequence[str], str] = "area",
        align_corners: Optional[bool] = None,
    ):
        super().__init__(keys)
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.resizer = Resize(spatial_size=spatial_size, align_corners=align_corners)

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.resizer(d[key], mode=self.mode[idx])
        return d


class RandAffined(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.RandAffine`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size,
        prob: float = 0.1,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        mode: Union[Sequence[str], str] = "bilinear",
        padding_mode: Union[Sequence[str], str] = "zeros",
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spatial_size (list or tuple of int): output image spatial size.
                if ``data`` component has two spatial dimensions, ``spatial_size`` should have 2 elements [h, w].
                if ``data`` component has three spatial dimensions, ``spatial_size`` should have 3 elements [h, w, d].
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                For a sequence each element corresponds to a key in ``keys``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                For a sequence each element corresponds to a key in ``keys``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.

        See also:
            - :py:class:`monai.transforms.compose.MapTransform`
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
        """
        super().__init__(keys)
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

    def set_random_state(self, seed=None, state=None):
        self.rand_affine.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self) -> None:  # type: ignore # see issue #495
        self.rand_affine.randomize()

    def __call__(self, data):
        d = dict(data)
        self.randomize()

        spatial_size = self.rand_affine.spatial_size
        if self.rand_affine.do_transform:
            grid = self.rand_affine.rand_affine_grid(spatial_size=spatial_size)
        else:
            grid = create_grid(spatial_size=spatial_size)

        for idx, key in enumerate(self.keys):
            d[key] = self.rand_affine.resampler(d[key], grid, mode=self.mode[idx], padding_mode=self.padding_mode[idx])
        return d


class Rand2DElasticd(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.Rand2DElastic`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        spatial_size,
        spacing,
        magnitude_range,
        prob: float = 0.1,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        mode: Union[Sequence[str], str] = "bilinear",
        padding_mode: Union[Sequence[str], str] = "zeros",
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spatial_size (2 ints): specifying output image spatial size [h, w].
            spacing (2 ints): distance in between the control points.
            magnitude_range (2 ints): the random offsets will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                For a sequence each element corresponds to a key in ``keys``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                For a sequence each element corresponds to a key in ``keys``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.
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

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        self.rand_2d_elastic.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, spatial_size) -> None:  # type: ignore # see issue #495
        self.rand_2d_elastic.randomize(spatial_size)

    def __call__(self, data):
        d = dict(data)
        spatial_size = self.rand_2d_elastic.spatial_size
        if np.any([sz <= 1 for sz in spatial_size]):
            spatial_size = data[self.keys[0]].shape[1:]
        self.randomize(spatial_size)

        if self.rand_2d_elastic.do_transform:
            grid = self.rand_2d_elastic.deform_grid(spatial_size)
            grid = self.rand_2d_elastic.rand_affine_grid(grid=grid)
            grid = _torch_interp(input=grid[None], size=spatial_size, mode="bicubic", align_corners=False)[0]
        else:
            grid = create_grid(spatial_size)

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
        spatial_size,
        sigma_range,
        magnitude_range,
        prob: float = 0.1,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        mode: Union[Sequence[str], str] = "bilinear",
        padding_mode: Union[Sequence[str], str] = "zeros",
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            keys: keys of the corresponding items to be transformed.
            spatial_size (3 ints): specifying output image spatial size [h, w, d].
            sigma_range (2 ints): a Gaussian kernel with standard deviation sampled
                 from ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range (2 ints): the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                For a sequence each element corresponds to a key in ``keys``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                For a sequence each element corresponds to a key in ``keys``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.
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

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        self.rand_3d_elastic.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, grid_size) -> None:  # type: ignore # see issue #495
        self.rand_3d_elastic.randomize(grid_size)

    def __call__(self, data):
        d = dict(data)
        spatial_size = self.rand_3d_elastic.spatial_size
        if np.any([sz <= 1 for sz in spatial_size]):
            spatial_size = data[self.keys[0]].shape[1:]
        self.randomize(spatial_size)
        grid = create_grid(spatial_size)
        if self.rand_3d_elastic.do_transform:
            device = self.rand_3d_elastic.device
            grid = torch.tensor(grid).to(device)
            gaussian = GaussianFilter(spatial_dims=3, sigma=self.rand_3d_elastic.sigma, truncated=3.0).to(device)
            offset = torch.tensor(self.rand_3d_elastic.rand_offset[None], device=device)
            grid[:3] += gaussian(offset)[0] * self.rand_3d_elastic.magnitude
            grid = self.rand_3d_elastic.rand_affine_grid(grid=grid)

        for idx, key in enumerate(self.keys):
            d[key] = self.rand_3d_elastic.resampler(
                d[key], grid, mode=self.mode[idx], padding_mode=self.padding_mode[idx]
            )
        return d


class Flipd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.Flip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        keys (dict): Keys to pick data for transformation.
        spatial_axis (None, int or tuple of ints): Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, keys: KeysCollection, spatial_axis=None):
        super().__init__(keys)
        self.flipper = Flip(spatial_axis=spatial_axis)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d


class RandFlipd(Randomizable, MapTransform):
    """Dictionary-based version :py:class:`monai.transforms.RandFlip`.

    See `numpy.flip` for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.
        spatial_axis (None, int or tuple of ints): Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, keys: KeysCollection, prob: float = 0.1, spatial_axis=None):
        super().__init__(keys)
        self.spatial_axis = spatial_axis
        self.prob = prob

        self._do_transform = False
        self.flipper = Flip(spatial_axis=spatial_axis)

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._do_transform = self.R.random_sample() < self.prob

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = self.flipper(d[key])
        return d


class Rotated(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.Rotate`.

    Args:
        keys (dict): Keys to pick data for transformation.
        angle (float or sequence of float): Rotation angle(s) in degrees.
        keep_size (bool): If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            For a sequence each element corresponds to a key in ``keys``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            For a sequence each element corresponds to a key in ``keys``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        align_corners (bool): Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    def __init__(
        self,
        keys: KeysCollection,
        angle,
        keep_size: bool = True,
        mode: Union[Sequence[str], str] = "bilinear",
        padding_mode: Union[Sequence[str], str] = "border",
        align_corners: bool = False,
    ):
        super().__init__(keys)
        self.rotator = Rotate(angle=angle, keep_size=keep_size, align_corners=align_corners)

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.rotator(d[key], mode=self.mode[idx], padding_mode=self.padding_mode[idx])
        return d


class RandRotated(Randomizable, MapTransform):
    """Dictionary-based version :py:class:`monai.transforms.RandRotate`
    Randomly rotates the input arrays.

    Args:
        range_x (tuple of float or float): Range of rotation angle in degrees in the
            plane defined by the first and second axes.
            If single number, angle is uniformly sampled from (-range_x, range_x).
        range_y (tuple of float or float): Range of rotation angle in degrees in the
            plane defined by the first and third axes.
            If single number, angle is uniformly sampled from (-range_y, range_y).
        range_z (tuple of float or float): Range of rotation angle in degrees in the
            plane defined by the second and third axes.
            If single number, angle is uniformly sampled from (-range_z, range_z).
        prob (float): Probability of rotation.
        keep_size (bool): If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            For a sequence each element corresponds to a key in ``keys``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            For a sequence each element corresponds to a key in ``keys``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        align_corners (bool): Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    def __init__(
        self,
        keys: KeysCollection,
        range_x=0.0,
        range_y=0.0,
        range_z=0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: Union[Sequence[str], str] = "bilinear",
        padding_mode: Union[Sequence[str], str] = "border",
        align_corners: bool = False,
    ):
        super().__init__(keys)
        self.range_x = ensure_tuple(range_x)
        if len(self.range_x) == 1:
            self.range_x = tuple(sorted([-self.range_x[0], self.range_x[0]]))
        self.range_y = ensure_tuple(range_y)
        if len(self.range_y) == 1:
            self.range_y = tuple(sorted([-self.range_y[0], self.range_y[0]]))
        self.range_z = ensure_tuple(range_z)
        if len(self.range_z) == 1:
            self.range_z = tuple(sorted([-self.range_z[0], self.range_z[0]]))

        self.prob = prob
        self.keep_size = keep_size
        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.padding_mode = ensure_tuple_rep(padding_mode, len(self.keys))
        self.align_corners = align_corners

        self._do_transform = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._do_transform = self.R.random_sample() < self.prob
        self.x = self.R.uniform(low=self.range_x[0], high=self.range_x[1])
        self.y = self.R.uniform(low=self.range_y[0], high=self.range_y[1])
        self.z = self.R.uniform(low=self.range_z[0], high=self.range_z[1])

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        rotator = Rotate(
            angle=self.x if d[self.keys[0]].ndim == 3 else (self.x, self.y, self.z),
            keep_size=self.keep_size,
            align_corners=self.align_corners,
        )
        for idx, key in enumerate(self.keys):
            d[key] = rotator(d[key], mode=self.mode[idx], padding_mode=self.padding_mode[idx])
        return d


class Zoomd(MapTransform):
    """Dictionary-based wrapper of :py:class:`monai.transforms.Zoom`.

    Args:
        zoom (float or sequence): The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            For a sequence each element corresponds to a key in ``keys``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        align_corners (optional bool): This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        keep_size (bool): Should keep original size (pad if needed), default is True.
    """

    def __init__(
        self,
        keys: KeysCollection,
        zoom,
        mode: Union[Sequence[str], str] = "area",
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
    ):
        super().__init__(keys)
        self.zoomer = Zoom(zoom=zoom, align_corners=align_corners, keep_size=keep_size)
        self.mode = ensure_tuple_rep(mode, len(self.keys))

    def __call__(self, data):
        d = dict(data)
        for idx, key in enumerate(self.keys):
            d[key] = self.zoomer(d[key], mode=self.mode[idx])
        return d


class RandZoomd(Randomizable, MapTransform):
    """Dict-based version :py:class:`monai.transforms.RandZoom`.

    Args:
        keys: Keys to pick data for transformation.
        prob: Probability of zooming.
        min_zoom (float or sequence): Min zoom factor. Can be float or sequence same size as image.
            If a float, min_zoom is the same for each spatial axis.
            If a sequence, min_zoom should contain one value for each spatial axis.
        max_zoom (float or sequence): Max zoom factor. Can be float or sequence same size as image.
            If a float, max_zoom is the same for each spatial axis.
            If a sequence, max_zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            For a sequence each element corresponds to a key in ``keys``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        align_corners (optional bool): This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        keep_size: Should keep original size (pad if needed), default is True.
    """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        min_zoom=0.9,
        max_zoom=1.1,
        mode: Union[Sequence[str], str] = "area",
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
    ):
        super().__init__(keys)
        if hasattr(min_zoom, "__iter__") and hasattr(max_zoom, "__iter__"):
            assert len(min_zoom) == len(max_zoom), "min_zoom and max_zoom must have same length."
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.prob = prob

        self.mode = ensure_tuple_rep(mode, len(self.keys))
        self.align_corners = align_corners
        self.keep_size = keep_size

        self._do_transform = False
        self._zoom: Optional[np.random.RandomState] = None

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._do_transform = self.R.random_sample() < self.prob
        if hasattr(self.min_zoom, "__iter__"):
            self._zoom = (self.R.uniform(l, h) for l, h in zip(self.min_zoom, self.max_zoom))
        else:
            self._zoom = self.R.uniform(self.min_zoom, self.max_zoom)

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        zoomer = Zoom(self._zoom, align_corners=self.align_corners, keep_size=self.keep_size)
        for idx, key in enumerate(self.keys):
            d[key] = zoomer(d[key], mode=self.mode[idx])
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
