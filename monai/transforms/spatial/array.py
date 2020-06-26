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
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import warnings
from typing import List, Optional, Union

import numpy as np
import torch

from monai.config import get_torch_version_tuple
from monai.data.utils import compute_shape_offset, to_affine_nd, zoom_affine
from monai.networks.layers import AffineTransform, GaussianFilter
from monai.transforms.compose import Randomizable, Transform
from monai.transforms.utils import (
    create_control_grid,
    create_grid,
    create_rotate,
    create_scale,
    create_shear,
    create_translate,
)
from monai.utils.misc import ensure_tuple, ensure_tuple_rep, ensure_tuple_size
from monai.utils import optional_import

nib, _ = optional_import("nibabel")

if get_torch_version_tuple() >= (1, 5):
    # additional argument since torch 1.5 (to avoid warnings)
    def _torch_interp(**kwargs):
        return torch.nn.functional.interpolate(recompute_scale_factor=False, **kwargs)


else:
    _torch_interp = torch.nn.functional.interpolate


class Spacing(Transform):
    """
    Resample input image into the specified `pixdim`.
    """

    def __init__(
        self,
        pixdim,
        diagonal: bool = False,
        mode: str = "bilinear",
        padding_mode: str = "border",
        dtype: Optional[np.dtype] = None,
    ):
        """
        Args:
            pixdim (sequence of floats): output voxel spacing.
            diagonal: whether to resample the input to have a diagonal affine matrix.
                If True, the input data is resampled to the following affine::

                    np.diag((pixdim_0, pixdim_1, ..., pixdim_n, 1))

                This effectively resets the volume to the world coordinate system (RAS+ in nibabel).
                The original orientation, rotation, shearing are not preserved.

                If False, this transform preserves the axes orientation, orthogonal rotation and
                translation components from the original affine. This option will not flip/swap axes
                of the original data.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"border"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            dtype (None or np.dtype): output array data type, defaults to np.float32.
        """
        self.pixdim = np.array(ensure_tuple(pixdim), dtype=np.float64)
        self.diagonal = diagonal
        self.mode = mode
        self.padding_mode = padding_mode
        self.dtype = dtype

    def __call__(
        self,
        data_array: np.ndarray,
        affine=None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
        dtype: Optional[np.dtype] = None,
    ):
        """
        Args:
            data_array (ndarray): in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        Returns:
            data_array (resampled into `self.pixdim`), original pixdim, current pixdim.
        """
        sr = data_array.ndim - 1
        if sr <= 0:
            raise ValueError("the array should have at least one spatial dimension.")
        if affine is None:
            # default to identity
            affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_ = to_affine_nd(sr, affine)
        out_d = self.pixdim[:sr]
        if out_d.size < sr:
            out_d = np.append(out_d, [1.0] * (out_d.size - sr))
        if np.any(out_d <= 0):
            raise ValueError(f"pixdim must be positive, got {out_d}")
        # compute output affine, shape and offset
        new_affine = zoom_affine(affine_, out_d, diagonal=self.diagonal)
        output_shape, offset = compute_shape_offset(data_array.shape[1:], affine_, new_affine)
        new_affine[:sr, -1] = offset[:sr]
        transform = np.linalg.inv(affine_) @ new_affine
        # adapt to the actual rank
        transform_ = to_affine_nd(sr, transform)
        _dtype = dtype or self.dtype or np.float32

        # no resampling if it's identity transform
        if np.allclose(transform_, np.diag(np.ones(len(transform_))), atol=1e-3):
            output_data = data_array.copy().astype(_dtype)
            new_affine = to_affine_nd(affine, new_affine)
            return output_data, affine, new_affine

        # resample
        affine_xform = AffineTransform(
            normalized=False,
            mode=mode or self.mode,
            padding_mode=padding_mode or self.padding_mode,
            align_corners=True,
            reverse_indexing=True,
        )
        output_data = affine_xform(
            torch.from_numpy((data_array.astype(np.float64))[None]),  # AffineTransform requires a batch dim
            torch.from_numpy(transform_.astype(np.float64)),
            spatial_size=output_shape,
        )
        output_data = output_data.squeeze(0).detach().cpu().numpy().astype(_dtype)
        new_affine = to_affine_nd(affine, new_affine)
        return output_data, affine, new_affine


class Orientation(Transform):
    """
    Change the input image's orientation into the specified based on `axcodes`.
    """

    def __init__(self, axcodes=None, as_closest_canonical: bool = False, labels=tuple(zip("LPI", "RAS"))):
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

        See Also: `nibabel.orientations.ornt2axcodes`.
        """
        if axcodes is None and not as_closest_canonical:
            raise ValueError("provide either `axcodes` or `as_closest_canonical=True`.")
        if axcodes is not None and as_closest_canonical:
            warnings.warn("using as_closest_canonical=True, axcodes ignored.")
        self.axcodes = axcodes
        self.as_closest_canonical = as_closest_canonical
        self.labels = labels

    def __call__(self, data_array: np.ndarray, affine=None):
        """
        original orientation of `data_array` is defined by `affine`.

        Args:
            data_array (ndarray): in shape (num_channels, H[, W, ...]).
            affine (matrix): (N+1)x(N+1) original affine matrix for spatially ND `data_array`. Defaults to identity.
        Returns:
            data_array (reoriented in `self.axcodes`), original axcodes, current axcodes.
        """
        sr = data_array.ndim - 1
        if sr <= 0:
            raise ValueError("the array should have at least one spatial dimension.")
        if affine is None:
            affine = np.eye(sr + 1, dtype=np.float64)
            affine_ = np.eye(sr + 1, dtype=np.float64)
        else:
            affine_ = to_affine_nd(sr, affine)
        src = nib.io_orientation(affine_)
        if self.as_closest_canonical:
            spatial_ornt = src
        else:
            dst = nib.orientations.axcodes2ornt(self.axcodes[:sr], labels=self.labels)
            if len(dst) < sr:
                raise ValueError(
                    f"`self.axcodes` should have at least {sr} elements"
                    f' given the data array is in spatial {sr}D, got "{self.axcodes}"'
                )
            spatial_ornt = nib.orientations.ornt_transform(src, dst)
        ornt = spatial_ornt.copy()
        ornt[:, 0] += 1  # skip channel dim
        ornt = np.concatenate([np.array([[0, 1]]), ornt])
        shape = data_array.shape[1:]
        data_array = nib.orientations.apply_orientation(data_array, ornt)
        new_affine = affine_ @ nib.orientations.inv_ornt_aff(spatial_ornt, shape)
        new_affine = to_affine_nd(affine, new_affine)
        return data_array, affine, new_affine


class Flip(Transform):
    """Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``np.flip`` in practice. See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        spatial_axis (None, int or tuple of ints): spatial axes along which to flip over. Default is None.
    """

    def __init__(self, spatial_axis=None) -> None:
        self.spatial_axis = spatial_axis

    def __call__(self, img):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        flipped = list()
        for channel in img:
            flipped.append(np.flip(channel, self.spatial_axis))
        return np.stack(flipped).astype(img.dtype)


class Resize(Transform):
    """
    Resize the input image to given spatial size.
    Implemented using :py:class:`torch.nn.functional.interpolate`.

    Args:
        spatial_size (tuple or list): expected shape of spatial dimensions after resize operation.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    def __init__(self, spatial_size, mode: str = "area", align_corners: Optional[bool] = None):
        self.spatial_size = ensure_tuple(spatial_size)
        self.mode = mode
        self.align_corners = align_corners

    def __call__(self, img, mode: Optional[str] = None):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
            mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
                The interpolation mode. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        """
        input_ndim = img.ndim - 1  # spatial ndim
        output_ndim = len(self.spatial_size)
        if output_ndim > input_ndim:
            input_shape = ensure_tuple_size(img.shape, output_ndim + 1, 1)
            img = img.reshape(input_shape)
        elif output_ndim < input_ndim:
            raise ValueError(
                "len(spatial_size) cannot be smaller than the image spatial dimensions, "
                f"got {output_ndim} and {input_ndim}."
            )
        resized = _torch_interp(
            input=torch.as_tensor(img[None], dtype=torch.float),
            size=self.spatial_size,
            mode=mode or self.mode,
            align_corners=self.align_corners,
        )
        resized = resized.squeeze(0).detach().cpu().numpy()
        return resized


class Rotate(Transform):
    """
    Rotates an input image by given angle using :py:class:`monai.networks.layers.AffineTransform`.

    Args:
        angle (float or sequence of float): Rotation angle(s) in degrees.
            should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    def __init__(
        self,
        angle,
        keep_size: bool = True,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
    ):
        self.angle = angle
        self.keep_size = keep_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def __call__(self, img, mode: Optional[str] = None, padding_mode: Optional[str] = None):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        im_shape = np.asarray(img.shape[1:])  # spatial dimensions
        input_ndim = len(im_shape)
        if input_ndim not in (2, 3):
            raise ValueError("Rotate only supports 2D and 3D: [chns, H, W] and [chns, H, W, D].")
        _angle = ensure_tuple_rep(self.angle, 1 if input_ndim == 2 else 3)
        _rad = np.deg2rad(_angle)
        transform = create_rotate(input_ndim, _rad)
        shift = create_translate(input_ndim, (im_shape - 1) / 2)
        if self.keep_size:
            output_shape = im_shape
        else:
            corners = np.asarray(np.meshgrid(*[(0, dim) for dim in im_shape], indexing="ij")).reshape(
                (len(im_shape), -1)
            )
            corners = transform[:-1, :-1] @ corners
            output_shape = (corners.ptp(axis=1) + 0.5).astype(int)
        shift_1 = create_translate(input_ndim, -(output_shape - 1) / 2)
        transform = shift @ transform @ shift_1
        _dtype = img.dtype
        xform = AffineTransform(
            normalized=False,
            mode=mode or self.mode,
            padding_mode=padding_mode or self.padding_mode,
            align_corners=self.align_corners,
            reverse_indexing=True,
        )
        output = xform(
            torch.from_numpy(img.astype(np.float64)[None]),
            torch.from_numpy(transform.astype(np.float64)),
            spatial_size=output_shape,
        )
        output = output.squeeze(0).detach().cpu().numpy().astype(_dtype)
        return output


class Zoom(Transform):
    """
    Zooms an ND image using :py:class:`torch.nn.functional.interpolate`.
    For details, please see https://pytorch.org/docs/stable/nn.functional.html#interpolate.

    Different from :py:class:`monai.transforms.resize`, this transform takes scaling factors
    as input, and provides an option of preserving the input spatial size.

    Args:
        zoom (float or sequence): The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        keep_size: Should keep original size (padding/slicing if needed), default is True.
    """

    def __init__(
        self, zoom, mode: str = "area", align_corners: Optional[bool] = None, keep_size: bool = True,
    ):
        self.zoom = zoom
        self.mode = mode
        self.align_corners = align_corners
        self.keep_size = keep_size

    def __call__(  # type: ignore # see issue #495
        self, img, mode: Optional[str] = None
    ):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        self.zoom = ensure_tuple_rep(self.zoom, img.ndim - 1)  # match the spatial image dim
        zoomed = _torch_interp(
            input=torch.as_tensor(img[None], dtype=torch.float),
            scale_factor=list(self.zoom),
            mode=mode or self.mode,
            align_corners=self.align_corners,
        )
        zoomed = zoomed.squeeze(0).detach().cpu().numpy()
        if not self.keep_size or np.allclose(img.shape, zoomed.shape):
            return zoomed

        pad_vec = [[0, 0]] * len(img.shape)
        slice_vec = [slice(None)] * len(img.shape)
        for idx, (od, zd) in enumerate(zip(img.shape, zoomed.shape)):
            diff = od - zd
            half = abs(diff) // 2
            if diff > 0:  # need padding
                pad_vec[idx] = [half, diff - half]
            elif diff < 0:  # need slicing
                slice_vec[idx] = slice(half, half + od)
        zoomed = np.pad(zoomed, pad_vec, mode="edge")
        return zoomed[tuple(slice_vec)]


class Rotate90(Transform):
    """
    Rotate an array by 90 degrees in the plane specified by `axes`.
    """

    def __init__(self, k: int = 1, spatial_axes=(0, 1)):
        """
        Args:
            k: number of times to rotate by 90 degrees.
            spatial_axes (2 ints): defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        self.k = k
        self.spatial_axes = spatial_axes

    def __call__(self, img: np.ndarray):
        """
        Args:
            img (ndarray): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        rotated = list()
        for channel in img:
            rotated.append(np.rot90(channel, self.k, self.spatial_axes))
        return np.stack(rotated).astype(img.dtype)


class RandRotate90(Randomizable, Transform):
    """
    With probability `prob`, input arrays are rotated by 90 degrees
    in the plane specified by `spatial_axes`.
    """

    def __init__(self, prob: float = 0.1, max_k: int = 3, spatial_axes=(0, 1)):
        """
        Args:
            prob: probability of rotating.
                (Default 0.1, with 10% probability it returns a rotated array)
            max_k: number of rotations will be sampled from `np.random.randint(max_k) + 1`.
                (Default 3)
            spatial_axes (2 ints): defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
        """
        self.prob = min(max(prob, 0.0), 1.0)
        self.max_k = max_k
        self.spatial_axes = spatial_axes

        self._do_transform = False
        self._rand_k = 0

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._rand_k = self.R.randint(self.max_k) + 1
        self._do_transform = self.R.random() < self.prob

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        rotator = Rotate90(self._rand_k, self.spatial_axes)
        return rotator(img)


class RandRotate(Randomizable, Transform):
    """
    Randomly rotate the input arrays.

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
        prob: Probability of rotation.
        keep_size: If it is False, the output shape is adapted so that the
            input array is contained completely in the output.
            If it is True, the output shape is the same as the input. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``"border"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        align_corners: Defaults to False.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
    """

    def __init__(
        self,
        range_x=0.0,
        range_y=0.0,
        range_z=0.0,
        prob: float = 0.1,
        keep_size: bool = True,
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = False,
    ):
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
        self.mode = mode
        self.padding_mode = padding_mode
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

    def __call__(self, img, mode: Optional[str] = None, padding_mode: Optional[str] = None):
        """
        Args:
            img (ndarray): channel first array, must have shape 2D: (nchannels, H, W), or 3D: (nchannels, H, W, D).
        """
        self.randomize()
        if not self._do_transform:
            return img
        rotator = Rotate(
            angle=self.x if img.ndim == 3 else (self.x, self.y, self.z),
            keep_size=self.keep_size,
            mode=mode or self.mode,
            padding_mode=padding_mode or self.padding_mode,
            align_corners=self.align_corners,
        )
        return rotator(img)


class RandFlip(Randomizable, Transform):
    """Randomly flips the image along axes. Preserves shape.
    See numpy.flip for additional details.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.flip.html

    Args:
        prob: Probability of flipping.
        spatial_axis (None, int or tuple of ints): Spatial axes along which to flip over. Default is None.
    """

    def __init__(self, prob: float = 0.1, spatial_axis=None):
        self.prob = prob
        self.flipper = Flip(spatial_axis=spatial_axis)
        self._do_transform = False

    def randomize(self) -> None:  # type: ignore # see issue #495
        self._do_transform = self.R.random_sample() < self.prob

    def __call__(self, img):
        self.randomize()
        if not self._do_transform:
            return img
        return self.flipper(img)


class RandZoom(Randomizable, Transform):
    """Randomly zooms input arrays with given probability within given zoom range.

    Args:
        prob: Probability of zooming.
        min_zoom (float or sequence): Min zoom factor. Can be float or sequence same size as image.
            If a float, min_zoom is the same for each spatial axis.
            If a sequence, min_zoom should contain one value for each spatial axis.
        max_zoom (float or sequence): Max zoom factor. Can be float or sequence same size as image.
            If a float, max_zoom is the same for each spatial axis.
            If a sequence, max_zoom should contain one value for each spatial axis.
        mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``"area"``.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        align_corners (optional bool): This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: None.
            See also: https://pytorch.org/docs/stable/nn.functional.html#interpolate
        keep_size (bool): Should keep original size (pad if needed), default is True.
    """

    def __init__(
        self,
        prob: float = 0.1,
        min_zoom=0.9,
        max_zoom=1.1,
        mode: str = "area",
        align_corners: Optional[bool] = None,
        keep_size: bool = True,
    ):
        if hasattr(min_zoom, "__iter__") and hasattr(max_zoom, "__iter__"):
            assert len(min_zoom) == len(max_zoom), "min_zoom and max_zoom must have same length."
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.prob = prob
        self.mode = mode
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

    def __call__(self, img, mode: Optional[str] = None):
        self.randomize()
        _dtype = np.float32
        if not self._do_transform:
            return img.astype(_dtype)
        zoomer = Zoom(self._zoom, align_corners=self.align_corners, keep_size=self.keep_size)
        return zoomer(img, mode=mode or self.mode).astype(_dtype)


class AffineGrid(Transform):
    """
    Affine transforms on the coordinates.
    """

    def __init__(
        self,
        rotate_params=None,
        shear_params=None,
        translate_params=None,
        scale_params=None,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.rotate_params = rotate_params
        self.shear_params = shear_params
        self.translate_params = translate_params
        self.scale_params = scale_params

        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(self, spatial_size=None, grid=None):
        """
        Args:
            spatial_size (list or tuple of int): output grid size.
            grid (ndarray): grid to be transformed. Shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
        """
        if grid is None:
            if spatial_size is not None:
                grid = create_grid(spatial_size)
            else:
                raise ValueError("Either specify a grid or a spatial size to create a grid from.")

        spatial_dims = len(grid.shape) - 1
        affine = np.eye(spatial_dims + 1)
        if self.rotate_params:
            affine = affine @ create_rotate(spatial_dims, self.rotate_params)
        if self.shear_params:
            affine = affine @ create_shear(spatial_dims, self.shear_params)
        if self.translate_params:
            affine = affine @ create_translate(spatial_dims, self.translate_params)
        if self.scale_params:
            affine = affine @ create_scale(spatial_dims, self.scale_params)
        affine = torch.as_tensor(np.ascontiguousarray(affine), device=self.device)

        grid = torch.tensor(grid) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            grid = grid.to(self.device)
        grid = (affine.float() @ grid.reshape((grid.shape[0], -1)).float()).reshape([-1] + list(grid.shape[1:]))
        if self.as_tensor_output:
            return grid
        return grid.cpu().numpy()


class RandAffineGrid(Randomizable, Transform):
    """
    generate randomised affine grid
    """

    def __init__(
        self,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            rotate_range (a sequence of positive floats): rotate_range[0] with be used to generate the 1st rotation
                parameter from `uniform[-rotate_range[0], rotate_range[0])`. Similarly, `rotate_range[2]` and
                `rotate_range[3]` are used in 3D affine for the range of 2nd and 3rd axes.
            shear_range (a sequence of positive floats): shear_range[0] with be used to generate the 1st shearing
                parameter from `uniform[-shear_range[0], shear_range[0])`. Similarly, `shear_range[1]` to
                `shear_range[N]` controls the range of the uniform distribution used to generate the 2nd to
                N-th parameter.
            translate_range (a sequence of positive floats): translate_range[0] with be used to generate the 1st
                shift parameter from `uniform[-translate_range[0], translate_range[0])`. Similarly, `translate_range[1]`
                to `translate_range[N]` controls the range of the uniform distribution used to generate
                the 2nd to N-th parameter.
            scale_range (a sequence of positive floats): scaling_range[0] with be used to generate the 1st scaling
                factor from `uniform[-scale_range[0], scale_range[0]) + 1.0`. Similarly, `scale_range[1]` to
                `scale_range[N]` controls the range of the uniform distribution used to generate the 2nd to
                N-th parameter.

        See also:
            - :py:meth:`monai.transforms.utils.create_rotate`
            - :py:meth:`monai.transforms.utils.create_shear`
            - :py:meth:`monai.transforms.utils.create_translate`
            - :py:meth:`monai.transforms.utils.create_scale`
        """
        self.rotate_range = ensure_tuple(rotate_range)
        self.shear_range = ensure_tuple(shear_range)
        self.translate_range = ensure_tuple(translate_range)
        self.scale_range = ensure_tuple(scale_range)

        self.rotate_params: Optional[List[float]] = None
        self.shear_params: Optional[List[float]] = None
        self.translate_params: Optional[List[float]] = None
        self.scale_params: Optional[List[float]] = None

        self.as_tensor_output = as_tensor_output
        self.device = device

    def randomize(self) -> None:  # type: ignore # see issue #495
        if self.rotate_range:
            self.rotate_params = [self.R.uniform(-f, f) for f in self.rotate_range if f is not None]
        if self.shear_range:
            self.shear_params = [self.R.uniform(-f, f) for f in self.shear_range if f is not None]
        if self.translate_range:
            self.translate_params = [self.R.uniform(-f, f) for f in self.translate_range if f is not None]
        if self.scale_range:
            self.scale_params = [self.R.uniform(-f, f) + 1.0 for f in self.scale_range if f is not None]

    def __call__(self, spatial_size=None, grid=None):
        """
        Returns:
            a 2D (3xHxW) or 3D (4xHxWxD) grid.
        """
        self.randomize()
        affine_grid = AffineGrid(
            rotate_params=self.rotate_params,
            shear_params=self.shear_params,
            translate_params=self.translate_params,
            scale_params=self.scale_params,
            as_tensor_output=self.as_tensor_output,
            device=self.device,
        )
        return affine_grid(spatial_size, grid)


class RandDeformGrid(Randomizable, Transform):
    """
    generate random deformation grid
    """

    def __init__(self, spacing, magnitude_range, as_tensor_output: bool = True, device: Optional[torch.device] = None):
        """
        Args:
            spacing (2 or 3 ints): spacing of the grid in 2D or 3D.
                e.g., spacing=(1, 1) indicates pixel-wise deformation in 2D,
                spacing=(1, 1, 1) indicates voxel-wise deformation in 3D,
                spacing=(2, 2) indicates deformation field defined on every other pixel in 2D.
            magnitude_range (2 ints): the random offsets will be generated from
                `uniform[magnitude[0], magnitude[1])`.
            as_tensor_output: whether to output tensor instead of numpy array.
                defaults to True.
            device (torch device): device to store the output grid data.
        """
        self.spacing = spacing
        self.magnitude = magnitude_range

        self.rand_mag = 1.0
        self.as_tensor_output = as_tensor_output
        self.random_offset = 0.0
        self.device = device

    def randomize(self, grid_size):
        self.random_offset = self.R.normal(size=([len(grid_size)] + list(grid_size))).astype(np.float32)
        self.rand_mag = self.R.uniform(self.magnitude[0], self.magnitude[1])

    def __call__(self, spatial_size):
        control_grid = create_control_grid(spatial_size, self.spacing)
        self.randomize(control_grid.shape[1:])
        control_grid[: len(spatial_size)] += self.rand_mag * self.random_offset
        if self.as_tensor_output:
            control_grid = torch.as_tensor(np.ascontiguousarray(control_grid), device=self.device)
        return control_grid


class Resample(Transform):
    def __init__(
        self,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        computes output image using values from `img`, locations from `grid` using pytorch.
        supports spatially 2D or 3D (num_channels, H, W[, D]).

        Args:
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: whether to return a torch tensor. Defaults to False.
            device (torch.device): device on which the tensor will be allocated.
        """
        self.mode = mode
        self.padding_mode = padding_mode
        self.as_tensor_output = as_tensor_output
        self.device = device

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        grid: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):
        """
        Args:
            img: shape must be (num_channels, H, W[, D]).
            grid: shape must be (3, H, W) for 2D or (4, H, W, D) for 3D.
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """

        if not torch.is_tensor(img):
            img = torch.as_tensor(np.ascontiguousarray(img))
        assert grid is not None, "Error, grid argument must be supplied as an ndarray or tensor "
        grid = torch.tensor(grid) if not torch.is_tensor(grid) else grid.detach().clone()
        if self.device:
            img = img.to(self.device)
            grid = grid.to(self.device)

        for i, dim in enumerate(img.shape[1:]):
            grid[i] = 2.0 * grid[i] / (dim - 1.0)
        grid = grid[:-1] / grid[-1:]
        index_ordering: List[int] = [x for x in range(img.ndim - 2, -1, -1)]
        grid = grid[index_ordering]
        grid = grid.permute(list(range(grid.ndim))[1:] + [0])
        out = torch.nn.functional.grid_sample(
            img[None].float(),
            grid[None].float(),
            mode=mode or self.mode,
            padding_mode=padding_mode or self.padding_mode,
            align_corners=False,
        )[0]
        if self.as_tensor_output:
            return out
        return out.cpu().numpy()


class Affine(Transform):
    """
    transform ``img`` given the affine parameters.
    """

    def __init__(
        self,
        rotate_params=None,
        shear_params=None,
        translate_params=None,
        scale_params=None,
        spatial_size=None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        The affine transformations are applied in rotate, shear, translate, scale order.

        Args:
            rotate_params (float, list of floats): a rotation angle in radians,
                a scalar for 2D image, a tuple of 3 floats for 3D. Defaults to no rotation.
            shear_params (list of floats):
                a tuple of 2 floats for 2D, a tuple of 6 floats for 3D. Defaults to no shearing.
            translate_params (list of floats):
                a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Translation is in pixel/voxel
                relative to the center of the input image. Defaults to no translation.
            scale_params (list of floats):
                a tuple of 2 floats for 2D, a tuple of 3 floats for 3D. Defaults to no scaling.
            spatial_size (list or tuple of int): output image spatial size.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.
        """
        self.affine_grid = AffineGrid(
            rotate_params=rotate_params,
            shear_params=shear_params,
            translate_params=translate_params,
            scale_params=scale_params,
            as_tensor_output=True,
            device=device,
        )
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode = padding_mode

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size=None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W[, D]),
            spatial_size (list or tuple of int): output image spatial size.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        grid = self.affine_grid(spatial_size=spatial_size or self.spatial_size)
        return self.resampler(
            img=img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )


class RandAffine(Randomizable, Transform):
    """
    Random affine transform.
    """

    def __init__(
        self,
        prob: float = 0.1,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        spatial_size=None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        as_tensor_output: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid.
            spatial_size (list or tuple of int): output image spatial size.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """

        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            as_tensor_output=True,
            device=device,
        )
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)

        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode = padding_mode

        self.do_transform = False
        self.prob = prob

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self) -> None:  # type: ignore # see issue #495
        self.do_transform = self.R.rand() < self.prob
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size=None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W[, D]),
            spatial_size (list or tuple of int): output image spatial size.
                if `img` has two spatial dimensions, `spatial_size` should have 2 elements [h, w].
                if `img` has three spatial dimensions, `spatial_size` should have 3 elements [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        self.randomize()
        _spatial_size = spatial_size or self.spatial_size
        if self.do_transform:
            grid = self.rand_affine_grid(spatial_size=_spatial_size)
        else:
            grid = create_grid(_spatial_size)
        return self.resampler(
            img=img, grid=grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode
        )


class Rand2DElastic(Randomizable, Transform):
    """
    Random elastic deformation and affine in 2D
    """

    def __init__(
        self,
        spacing,
        magnitude_range,
        prob: float = 0.1,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        spatial_size=None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            spacing (2 ints): distance in between the control points.
            magnitude_range (2 ints): the random offsets will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            spatial_size (2 ints): specifying output image spatial size [h, w].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        self.deform_grid = RandDeformGrid(
            spacing=spacing, magnitude_range=magnitude_range, as_tensor_output=True, device=device
        )
        self.rand_affine_grid = RandAffineGrid(
            rotate_range=rotate_range,
            shear_range=shear_range,
            translate_range=translate_range,
            scale_range=scale_range,
            as_tensor_output=True,
            device=device,
        )
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)

        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.prob = prob
        self.do_transform = False

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        self.deform_grid.set_random_state(seed, state)
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, spatial_size):
        self.do_transform = self.R.rand() < self.prob
        self.deform_grid.randomize(spatial_size)
        self.rand_affine_grid.randomize()

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
        spatial_size=None,
        mode: Optional[str] = None,
        padding_mode: Optional[str] = None,
    ):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W),
            spatial_size (2 ints): specifying output image spatial size [h, w].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        spatial_size = spatial_size or self.spatial_size
        self.randomize(spatial_size)
        if self.do_transform:
            grid = self.deform_grid(spatial_size=spatial_size)
            grid = self.rand_affine_grid(grid=grid)
            grid = _torch_interp(input=grid[None], size=spatial_size, mode="bicubic", align_corners=False)[0]
        else:
            grid = create_grid(spatial_size)
        return self.resampler(img, grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode)


class Rand3DElastic(Randomizable, Transform):
    """
    Random elastic deformation and affine in 3D
    """

    def __init__(
        self,
        sigma_range,
        magnitude_range,
        prob: float = 0.1,
        rotate_range=None,
        shear_range=None,
        translate_range=None,
        scale_range=None,
        spatial_size=None,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        as_tensor_output: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            sigma_range (2 ints): a Gaussian kernel with standard deviation sampled
                 from ``uniform[sigma_range[0], sigma_range[1])`` will be used to smooth the random offset grid.
            magnitude_range (2 ints): the random offsets on the grid will be generated from
                ``uniform[magnitude[0], magnitude[1])``.
            prob: probability of returning a randomized affine grid.
                defaults to 0.1, with 10% chance returns a randomized grid,
                otherwise returns a ``spatial_size`` centered area extracted from the input image.
            spatial_size (3 ints): specifying output image spatial size [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``"bilinear"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``"zeros"``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            as_tensor_output: the computation is implemented using pytorch tensors, this option specifies
                whether to convert it back to numpy arrays.
            device (torch.device): device on which the tensor will be allocated.

        See also:
            - :py:class:`RandAffineGrid` for the random affine parameters configurations.
            - :py:class:`Affine` for the affine transformation parameters configurations.
        """
        self.rand_affine_grid = RandAffineGrid(rotate_range, shear_range, translate_range, scale_range, True, device)
        self.resampler = Resample(as_tensor_output=as_tensor_output, device=device)

        self.sigma_range = sigma_range
        self.magnitude_range = magnitude_range
        self.spatial_size = spatial_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.device = device

        self.prob = prob
        self.do_transform = False
        self.rand_offset = None
        self.magnitude = 1.0
        self.sigma = 1.0

    def set_random_state(self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None):
        self.rand_affine_grid.set_random_state(seed, state)
        super().set_random_state(seed, state)
        return self

    def randomize(self, grid_size):
        self.do_transform = self.R.rand() < self.prob
        if self.do_transform:
            self.rand_offset = self.R.uniform(-1.0, 1.0, [3] + list(grid_size)).astype(np.float32)
        self.magnitude = self.R.uniform(self.magnitude_range[0], self.magnitude_range[1])
        self.sigma = self.R.uniform(self.sigma_range[0], self.sigma_range[1])
        self.rand_affine_grid.randomize()

    def __call__(
        self, img, spatial_size=None, mode: Optional[str] = None, padding_mode: Optional[str] = None,
    ):
        """
        Args:
            img (ndarray or tensor): shape must be (num_channels, H, W, D),
            spatial_size (3 ints): specifying spatial 3D output image spatial size [h, w, d].
            mode: {``"bilinear"``, ``"nearest"``}
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        """
        spatial_size = spatial_size or self.spatial_size
        self.randomize(spatial_size)
        grid = create_grid(spatial_size)
        if self.do_transform:
            assert self.rand_offset is not None
            grid = torch.as_tensor(np.ascontiguousarray(grid), device=self.device)
            gaussian = GaussianFilter(3, self.sigma, 3.0).to(device=self.device)
            offset = torch.as_tensor(self.rand_offset[None], device=self.device)
            grid[:3] += gaussian(offset)[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        return self.resampler(img, grid, mode=mode or self.mode, padding_mode=padding_mode or self.padding_mode)
