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
from enum import Enum

import numpy as np
import torch

import monai
from monai.config import NdarrayOrTensor
from monai.data.utils import AFFINE_TOL
from monai.transforms.utils_pytorch_numpy_unification import allclose
from monai.utils import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    LazyAttr,
    NdimageMode,
    TraceKeys,
    convert_to_numpy,
    convert_to_tensor,
    look_up_option,
)

__all__ = ["resample", "combine_transforms"]


class Affine:
    """A class to represent an affine transform matrix."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_affine_shaped(data):
        """Check if the data is an affine matrix."""
        if isinstance(data, Affine):
            return True
        if isinstance(data, DisplacementField):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 2:
            return False
        return data.shape[-1] in (3, 4) and data.shape[-1] == data.shape[-2]


class DisplacementField:
    """A class to represent a dense displacement field."""

    __slots__ = ("data",)

    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_ddf_shaped(data):
        """Check if the data is a DDF."""
        if isinstance(data, DisplacementField):
            return True
        if isinstance(data, Affine):
            return False
        if not hasattr(data, "shape") or len(data.shape) < 3:
            return False
        return not Affine.is_affine_shaped(data)


def combine_transforms(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Given transforms A and B to be applied to x, return the combined transform (AB), so that A(B(x)) becomes AB(x)"""
    if Affine.is_affine_shaped(left) and Affine.is_affine_shaped(right):  # linear transforms
        left = convert_to_tensor(left.data if isinstance(left, Affine) else left, wrap_sequence=True)
        right = convert_to_tensor(right.data if isinstance(right, Affine) else right, wrap_sequence=True)
        return torch.matmul(left, right)
    if DisplacementField.is_ddf_shaped(left) and DisplacementField.is_ddf_shaped(
        right
    ):  # adds DDFs, do we need metadata if metatensor input?
        left = convert_to_tensor(left.data if isinstance(left, DisplacementField) else left, wrap_sequence=True)
        right = convert_to_tensor(right.data if isinstance(right, DisplacementField) else right, wrap_sequence=True)
        return left + right
    raise NotImplementedError


def affine_from_pending(pending_item):
    """Extract the affine matrix from a pending transform item."""
    if isinstance(pending_item, (torch.Tensor, np.ndarray)):
        return pending_item
    if isinstance(pending_item, dict):
        return pending_item[LazyAttr.AFFINE]
    return pending_item


def _lazy_interp_mode(mode):
    """
    Convert an interpolation mode recorded by a spatial transform into one usable by the lazy resampler
    (:py:class:`monai.transforms.SpatialResample`).

    ``"nearest-exact"`` is mapped to grid-sample ``"nearest"``, which samples at the same (half-pixel aligned)
    source locations; results can differ only for source locations exactly halfway between two voxels, where
    grid-sample rounds half to even and ``torch.nn.functional.interpolate`` rounds half up.

    Args:
        mode: the interpolation mode recorded in a pending operation's ``extra_info``, e.g. a
            :py:class:`monai.utils.GridSampleMode` or :py:class:`monai.utils.InterpolateMode` value, its string
            form, or a spline interpolation order (``0``-``5``).

    Returns:
        A grid-sample mode string or a spline order supported by the lazy resampler, or ``None`` if ``mode`` is
        unset or has no equivalent (e.g. ``"area"``), in which case the resampler's default mode is used.
    """
    if isinstance(mode, Enum):
        mode = mode.value
    if isinstance(mode, (int, np.integer)) and not isinstance(mode, bool):
        return int(mode) if int(mode) in range(6) else None  # spline order
    if not isinstance(mode, str):
        return None
    if mode in {m.value for m in GridSampleMode}:
        return mode
    if mode == InterpolateMode.NEAREST_EXACT:
        return GridSampleMode.NEAREST.value
    if mode.endswith("linear"):  # linear, bilinear, trilinear
        return GridSampleMode.BILINEAR.value
    return None


def _lazy_padding_mode(padding_mode):
    """
    Convert a padding mode recorded by a spatial transform into one usable by the lazy resampler
    (:py:class:`monai.transforms.SpatialResample`).

    Args:
        padding_mode: the padding mode recorded in a pending operation's ``extra_info``, e.g. a
            :py:class:`monai.utils.GridSamplePadMode` or :py:class:`monai.utils.NdimageMode` value or its string form.

    Returns:
        ``padding_mode`` as a string if it is a supported grid-sample or ndimage padding mode, otherwise ``None``,
        in which case the resampler's default padding mode is used.
    """
    if isinstance(padding_mode, Enum):
        padding_mode = padding_mode.value
    valid_modes = {*(m.value for m in GridSamplePadMode), *(m.value for m in NdimageMode)}
    if isinstance(padding_mode, str) and padding_mode in valid_modes:
        return padding_mode
    return None


def kwargs_from_pending(pending_item):
    """Extract kwargs from a pending transform item.

    When ``pending_item`` is a dict, ``align_corners`` is also extracted from its ``extra_info`` entry
    (if present and boolean) so the lazy pipeline preserves the original transform's alignment.
    Likewise, if the interpolation/padding modes are not set explicitly on the pending item, the ``mode``
    and ``padding_mode`` recorded in ``extra_info`` by the spatial transform (e.g. ``Spacing(mode="nearest")``)
    are used, so that lazy resampling honours the mode requested by the user.

    Args:
        pending_item: a pending operation, typically a dict pushed by a lazy transform (see
            :py:meth:`monai.data.MetaTensor.push_pending_operation`). Non-dict items (e.g. a bare affine
            matrix) carry no kwargs.

    Returns:
        A dict of resampling kwargs keyed by :py:class:`monai.utils.LazyAttr` values. For a dict
        ``pending_item`` it always contains ``LazyAttr.INTERP_MODE`` and ``LazyAttr.PADDING_MODE`` (``None`` if
        unset or unsupported), plus ``LazyAttr.SHAPE``, ``LazyAttr.DTYPE`` and ``LazyAttr.ALIGN_CORNERS`` when
        available. An empty dict is returned for non-dict items.
    """
    if not isinstance(pending_item, dict):
        return {}
    extra_info = pending_item.get(TraceKeys.EXTRA_INFO)
    extra_info = extra_info if isinstance(extra_info, dict) else {}
    interp_mode = pending_item.get(LazyAttr.INTERP_MODE, None)
    if interp_mode is None:
        interp_mode = _lazy_interp_mode(extra_info.get("mode"))
    padding_mode = pending_item.get(LazyAttr.PADDING_MODE, None)
    if padding_mode is None:
        padding_mode = _lazy_padding_mode(extra_info.get("padding_mode"))
    ret = {LazyAttr.INTERP_MODE: interp_mode, LazyAttr.PADDING_MODE: padding_mode}
    if LazyAttr.SHAPE in pending_item:
        ret[LazyAttr.SHAPE] = pending_item[LazyAttr.SHAPE]
    if LazyAttr.DTYPE in pending_item:
        ret[LazyAttr.DTYPE] = pending_item[LazyAttr.DTYPE]
    # Extract align_corners from extra_info if available
    if "align_corners" in extra_info:
        align_corners_val = extra_info["align_corners"]
        if isinstance(align_corners_val, bool):
            ret[LazyAttr.ALIGN_CORNERS] = align_corners_val
    return ret


def is_compatible_apply_kwargs(kwargs_1, kwargs_2):
    """Check if two sets of kwargs are compatible (to be combined in `apply`)."""
    return True


def requires_interp(matrix, atol=AFFINE_TOL):
    """
    Check whether the transformation matrix suggests voxel-wise interpolation.

    Returns None if the affine matrix suggests interpolation.
    Otherwise, the matrix suggests that the resampling could be achieved by simple array operations
    such as flip/permute/pad_nd/slice; in this case this function returns axes information about simple axes
    operations.

    Args:
        matrix: the affine matrix to check.
        atol: absolute tolerance for checking if the matrix is close to an integer.
    """
    matrix = convert_to_numpy(matrix, wrap_sequence=True)
    s = matrix[:, -1]
    if not np.allclose(s, np.round(s), atol=atol):
        return None

    ndim = len(matrix) - 1
    ox, oy = [], [0]
    for x, r in enumerate(matrix[:ndim, :ndim]):
        for y, c in enumerate(r):
            if np.isclose(c, -1, atol=atol) or np.isclose(c, 1, atol=atol):
                y_channel = y + 1  # the returned axis index starting with channel dim
                if x in ox or y_channel in oy:
                    return None
                ox.append(x)
                oy.append(y_channel)
            elif not np.isclose(c, 0.0, atol=atol):
                return None
    return oy


__override_lazy_keywords = {*list(LazyAttr), "atol"}


def resample(data: torch.Tensor, matrix: NdarrayOrTensor, kwargs: dict | None = None):
    """
    Resample `data` using the affine transformation defined by ``matrix``.

    Args:
        data: input data to be resampled.
        matrix: affine transformation matrix.
        kwargs: currently supports (see also: ``monai.utils.enums.LazyAttr``)

            - "lazy_shape" for output spatial shape
            - "lazy_padding_mode"
            - "lazy_interpolation_mode" (this option might be ignored when ``mode="auto"``.)
            - "lazy_align_corners"
            - "lazy_dtype" (dtype for resampling computation; this might be ignored when ``mode="auto"``.)
            - "atol" for tolerance for matrix floating point comparison.
            - "lazy_resample_mode" for resampling backend, default to `"auto"`. Setting to other values will use the
              `monai.transforms.SpatialResample` for resampling.

    See Also:
        :py:class:`monai.transforms.SpatialResample`
    """
    if not Affine.is_affine_shaped(matrix):
        raise NotImplementedError(f"Calling the dense grid resample API directly not implemented, {matrix.shape}.")
    if isinstance(data, monai.data.MetaTensor) and data.pending_operations:
        warnings.warn("data.pending_operations is not empty, the resampling output may be incorrect.")
    kwargs = kwargs or {}
    for k in kwargs:
        look_up_option(k, __override_lazy_keywords)
    atol = kwargs.get("atol", AFFINE_TOL)
    mode = kwargs.get(LazyAttr.RESAMPLE_MODE, "auto")

    init_kwargs = {
        "dtype": kwargs.get(LazyAttr.DTYPE, data.dtype),
        "align_corners": kwargs.get(LazyAttr.ALIGN_CORNERS, False),
    }
    ndim = len(matrix) - 1
    img = convert_to_tensor(data=data, track_meta=monai.data.get_track_meta())
    init_affine = monai.data.to_affine_nd(ndim, img.affine)
    spatial_size = kwargs.get(LazyAttr.SHAPE, None)
    out_spatial_size = img.peek_pending_shape() if spatial_size is None else spatial_size
    out_spatial_size = convert_to_numpy(out_spatial_size, wrap_sequence=True)
    call_kwargs = {
        "spatial_size": out_spatial_size,
        "dst_affine": init_affine @ monai.utils.convert_to_dst_type(matrix, init_affine)[0],
        "mode": kwargs.get(LazyAttr.INTERP_MODE),
        "padding_mode": kwargs.get(LazyAttr.PADDING_MODE),
    }

    axes = requires_interp(matrix, atol=atol)
    if axes is not None and mode == "auto" and not init_kwargs["align_corners"]:
        matrix_np = np.round(convert_to_numpy(matrix, wrap_sequence=True))
        full_transpose = np.argsort(axes).tolist()
        if not np.allclose(full_transpose, np.arange(len(full_transpose))):
            img = img.permute(full_transpose[: len(img.shape)])
        in_shape = img.shape[1 : ndim + 1]  # requires that ``img`` has empty pending operations
        matrix_np[:ndim] = matrix_np[[x - 1 for x in full_transpose[1:]]]
        flip = [idx + 1 for idx, val in enumerate(matrix_np[:ndim]) if val[idx] == -1]
        if flip:
            img = torch.flip(img, dims=flip)  # todo: if on cpu, using the np.flip is faster?
            for f in flip:
                ind_f = f - 1
                matrix_np[ind_f, ind_f] = 1
                matrix_np[ind_f, -1] = in_shape[ind_f] - 1 - matrix_np[ind_f, -1]
        if not np.all(out_spatial_size > 0):
            raise ValueError(f"Resampling out_spatial_size should be positive, got {out_spatial_size}.")
        if (
            allclose(matrix_np, np.eye(len(matrix_np)), atol=atol)
            and len(in_shape) == len(out_spatial_size)
            and allclose(convert_to_numpy(in_shape, wrap_sequence=True), out_spatial_size)
        ):
            img.affine = call_kwargs["dst_affine"]
            img = img.to(torch.float32)  # consistent with monai.transforms.spatial.functional.spatial_resample
            return img
        # pyrefly: ignore [bad-argument-type, implicit-import]
        img = monai.transforms.crop_or_pad_nd(img, matrix_np, out_spatial_size, mode=call_kwargs["padding_mode"])
        img = img.to(torch.float32)  # consistent with monai.transforms.spatial.functional.spatial_resample
        img.affine = call_kwargs["dst_affine"]
        return img

    # pyrefly: ignore [bad-argument-type, implicit-import]
    resampler = monai.transforms.SpatialResample(**init_kwargs)
    resampler.lazy = False  # resampler is a lazytransform
    with resampler.trace_transform(False):  # don't track this transform in `img`
        return resampler(img=img, **call_kwargs)
