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

from typing import Optional, Sequence, Tuple, Union

import numpy as np

import torch

from monai.networks.layers import GaussianFilter


from monai.transforms.utils import (
    create_grid,
    compatible_translate, create_scale, create_rotate, create_identity, create_rotate_90, create_flip,
)
from monai.config import DtypeLike
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.lazy.functional import (
    apply_align_corners,
    extents_from_shape,
    shape_from_extents, lazily_apply_op,
)
from monai.transforms.lazy.functional import MetaMatrix, is_matrix_shaped
from monai.utils import (
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    get_equivalent_dtype,
    look_up_option,
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode, LazyAttr,
)


# TODO: overriding of the operation name in the case that the function is being called from a random array / dict transform


def transform_shape(input_shape: Sequence[int], matrix: torch.Tensor):
    """
    TODO: this method should accept Matrix and Grid types also
    TODO: this method should be moved to transforms.utils
    Transform `input_shape` according to `transform`. This can be used for any transforms that
    widen / narrow the resulting region of interest (typically transforms that have a 'keep_size'
    parameter such as rotate.

    Args:
        input_shape: the shape to be transformed
        matrix: the matrix to apply to it

    Returns:
        The resulting shape
    """
    if not is_matrix_shaped(matrix):
        raise ValueError("'matrix' must have a valid 2d or 3d homogenous matrix shape but has shape "
                         f"{matrix.shape}")
    im_extents = extents_from_shape(input_shape, matrix.dtype)
    im_extents = [matrix @ e for e in im_extents]
    output_shape = shape_from_extents(input_shape, im_extents)
    return output_shape


def get_input_shape_and_dtype(shape_override, dtype_override, img):
    # if shape_override is set, it always wins
    input_shape = shape_override
    input_dtype = dtype_override

    if input_shape is None:
        if isinstance(img, MetaTensor) and len(img.pending_operations) > 0:
            input_shape = img.peek_pending_shape()
        else:
            input_shape = img.shape
    if input_dtype is None:
        if isinstance(img, MetaTensor) and len(img.pending_operations) > 0:
            input_dtype = img.peek_pending_dtype()
        else:
            input_dtype = img.dtype
    return input_shape, input_dtype


def identity(
    img: torch.Tensor,
    mode: Optional[Union[InterpolateMode, str]] = None,
    padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = None,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
    shape_override: Optional[Sequence[int]] = None,
    dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
    lazy_evaluation: Optional[bool] = True
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    mode_ = None if mode is None else look_up_option(mode, GridSampleMode)
    padding_mode_ = None if padding_mode is None else look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img_.dtype, torch.Tensor)

    transform = create_identity(input_ndim)

    metadata = {
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: input_shape,
        LazyAttr.OUT_DTYPE: dtype_,
    }
    if mode_ is not None:
        metadata[LazyAttr.INTERP_MODE] = mode_
    if padding_mode_ is not None:
        metadata[LazyAttr.PADDING_MODE] = padding_mode_
    # metadata[LazyAttr.DTYPE] = dtype_

    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)


def spacing(
    img: torch.Tensor,
    pixdim: Union[Sequence[float], float],
    # src_pixdim: Optional[Union[Sequence[float], float]] = None,
    diagonal: Optional[bool] = False,
    mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
    padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
    align_corners: Optional[bool] = False,
    dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
    scale_extent: bool = False,
    recompute_affine: bool = False,
    min_pixdim: Sequence[float] | float | np.ndarray | None = None,
    max_pixdim: Sequence[float] | float | np.ndarray | None = None,
    shape_override: Optional[Sequence[int]] = None,
    dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
    lazy_evaluation: Optional[bool] = True
):
    """
    TODO: spacing needs to updated to match the functionality of Spacing on dev
    Args:
        img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        anti_aliasing: bool, optional
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.

    Raises:
        ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    # src_pixdim_ = src_pixdim or img_.pixdim
    src_pixdim_ = img_.pixdim

    input_ndim = len(input_shape) - 1

    pixdim_ = ensure_tuple_rep(pixdim, input_ndim)
    src_pixdim_ = ensure_tuple_rep(src_pixdim_[:input_ndim], input_ndim)

    if diagonal is True:
        raise ValueError("'diagonal' value of True is not currently supported")

    mode_ = look_up_option(mode, GridSampleMode)
    padding_mode_ = look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img.dtype, torch.Tensor)
    zoom_factors = [j / i for i, j in zip(src_pixdim_, pixdim_)]
    shape_zoom_factors = [i / j for i, j in zip(src_pixdim_, pixdim_)]

    transform = create_scale(input_ndim, zoom_factors)
    shape_transform = create_scale(input_ndim, shape_zoom_factors)

    output_shape = transform_shape(input_shape, shape_transform)


    metadata = {
        "op": "spacing",
        "pixdim": pixdim_,
        "src_pixdim": src_pixdim_,
        "diagonal": diagonal,
        LazyAttr.INTERP_MODE: mode_,
        LazyAttr.PADDING_MODE: padding_mode_,
        LazyAttr.ALIGN_CORNERS: align_corners,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.OUT_DTYPE: dtype_,
        LazyAttr.OUT_SHAPE: output_shape
    }
    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)


# def orientation(
#         img: torch.Tensor,
#         axcodes: Optional[str] = None,
#         as_closest_canonical: Optional[bool] = False,
#         labels: Optional[Sequence[Tuple[str, str]]] = (("L", "R"), ("P", "A"), ("I", "S")),
#         src_affine: Optional[torch.Tensor] = None,
#         shape_override: Optional[Sequence] = None,
#         lazy_evaluation: Optional[bool] = True
# ):
#     """
#     Change the current transform for the image to the orientation specified by the `axcodes` parameter.
#     The precise operation depending upon the following:
#      - if `src_affine` is set, all other affine sources are ignored
#      - if `src_affine` is not set:
#        - if `img` is a metatensor:
#          - if `img` and has pending transforms, the affine from which to perform the operation is
#          calculated from its `.affine` property followed by the concatenated pending transforms
#          - if `img` does not have any pending transforms, the affine from which to perform the operation
#          is calculated from its `.affine` property
#        - if `img` is not a metatensor, identity is used
#     """
#     img_ = convert_to_tensor(img, track_meta=get_track_meta())
#
#     # if shape_override is set, it always wins
#     input_shape = shape_override
#
#     if input_shape is None:
#         if isinstance(img_, MetaTensor) and len(img_.pending_operations) > 0:
#             input_shape = img_.peek_pending_shape()
#         else:
#             input_shape = img_.shape
#
#     if axcodes is None and not as_closest_canonical:
#         raise ValueError("Incompatible values: axcodes=None and as_closest_canonical=True.")
#     if axcodes is not None and as_closest_canonical:
#         warnings.warn("using as_closest_canonical=True, axcodes ignored.")
#
#     spatial_dims = len(input_shape) - 1
#     if spatial_dims < 1:
#         raise ValueError("'img' must have at least one spatial dimensions")
#
#     # calculating the transform to be applied
#     # A: accumulated transform
#     # C: resulting transform
#     # B: transform to calculate
#     # AB = C
#     # A`AB = A`C  - multiply both sides by A`
#     # IB = A`C    - A'A = I
#     # B = A`C
#     result = torch.linalg.inv(src_affine) @ dest_affine


def flip(
        img: torch.Tensor,
        spatial_axis: Union[Sequence[int], int],
        shape_override: Optional[Sequence] = None,
        dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
        lazy_evaluation: Optional[bool] = True
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    spatial_axis_ = spatial_axis
    if spatial_axis_ is None:
        spatial_axis_ = tuple(i for i in range(len(input_shape[1:])))

    transform = create_flip(input_ndim, spatial_axis_)

    metadata = {
        "op": "flip",
        "spatial_axis": spatial_axis_,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: input_shape,
        LazyAttr.OUT_DTYPE: input_dtype,
    }
    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)


def resize(
        img: torch.Tensor,
        spatial_size: Union[Sequence[int], int],
        size_mode: str = "all",
        mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
        align_corners: Optional[bool] = False,
        anti_aliasing: Optional[bool] = None,
        anti_aliasing_sigma: Optional[Union[Sequence[float], float]] = None,
        dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
        shape_override: Optional[Sequence[int]] = None,
        dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
        lazy_evaluation: Optional[bool] = True
):
    """
    Args:
        dtype_override:
        img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        anti_aliasing: bool, optional
            Whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
            By default, this value is chosen as (s - 1) / 2 where s is the
            downsampling factor, where s > 1. For the up-size case, s < 1, no
            anti-aliasing is performed prior to rescaling.

    Raises:
        ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    if size_mode == "all":
        spatial_size_ = fall_back_tuple(spatial_size, input_shape[1:])
        output_ndim = len(ensure_tuple(spatial_size_))
        if output_ndim > input_ndim:
            input_shape = ensure_tuple_size(input_shape, output_ndim + 1, 1)
            img = img.reshape(input_shape)
        elif output_ndim < input_ndim:
            raise ValueError(
                "len(spatial_size) must be greater or equal to img spatial dimensions, "
                f"got spatial_size={output_ndim} img={input_ndim}."
            )
    else:  # for the "longest" mode
        img_size = input_shape[1:]
        if not isinstance(spatial_size, int):
            raise ValueError("spatial_size must be an int number if size_mode is 'longest'.")
        scale = spatial_size / max(img_size)
        spatial_size_ = tuple(int(round(s * scale)) for s in img_size)

    mode_ = look_up_option(mode, InterpolateMode)
    dtype_ = get_equivalent_dtype(dtype or img.dtype, torch.Tensor)
    shape_zoom_factors = [i / j for i, j in zip(spatial_size_, input_shape[1:])]
    pixel_zoom_factors = [j / i for i, j in zip(spatial_size_, input_shape[1:])]
    print("zoom_factors:", shape_zoom_factors)

    shape_transform = create_scale(input_ndim, shape_zoom_factors)
    pixel_transform = create_scale(input_ndim, pixel_zoom_factors)
    print("transform:", shape_transform)

    output_shape = transform_shape(input_shape, shape_transform)
    print("output_shape:", output_shape)

    metadata = {
        "op": "resize",
        "spatial_size": spatial_size,
        "size_mode": size_mode,
        LazyAttr.INTERP_MODE: mode_,
        LazyAttr.ALIGN_CORNERS: align_corners,
        "anti_aliasing": anti_aliasing,
        "anti_aliasing_sigma": anti_aliasing_sigma,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: output_shape,
        LazyAttr.OUT_DTYPE: dtype_,
    }
    return lazily_apply_op(img_, MetaMatrix(pixel_transform, metadata), lazy_evaluation)


def rotate(
        img: torch.Tensor,
        angle: Union[Sequence[float], float],
        keep_size: Optional[bool] = True,
        mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.AREA,
        padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = False,
        dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
        shape_override: Optional[Sequence[int]] = None,
        dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
        lazy_evaluation: Optional[bool] = True
):
    """
    Args:
        img: channel first array, must have shape: [chns, H, W] or [chns, H, W, D].
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        keep_size: If it is True, the output shape is kept the same as the input.
            If it is False, the output shape is adapted so that the
            input array is contained completely in the output. Default is True.
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values. Defaults to ``self.padding_mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            align_corners: Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation. Defaults to ``self.dtype``.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``np.float32``.

    Raises:
        ValueError: When ``img`` spatially is not one of [2D, 3D].

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())
    mode_ = look_up_option(mode, GridSampleMode)
    padding_mode_ = look_up_option(padding_mode, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img_.dtype, torch.Tensor)
    # img_ = img_.to(dtype_)

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1
    if input_ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimension: {input_ndim}, available options are [2, 3].")

    angle_ = ensure_tuple_rep(angle, 1 if input_ndim == 2 else 3)

    # rotate_tx = compatible_rotate(img_, angle_)
    rotate_tx = create_rotate(input_ndim, angle_).astype(np.float64)
    output_shape = input_shape if keep_size is True else transform_shape(input_shape, rotate_tx)

    if align_corners is True:
        # op = lambda scale_factors: compatible_scale(img_, scale_factors)
        op = lambda scale_factors: create_scale(input_ndim, scale_factors).astype(np.float64)
        transform = apply_align_corners(rotate_tx, output_shape[1:], op)
    else:
        transform = rotate_tx

    metadata = {
        "op": "rotate",
        "angle": angle,
        "keep_size": keep_size,
        LazyAttr.INTERP_MODE: mode_,
        LazyAttr.PADDING_MODE: padding_mode_,
        LazyAttr.ALIGN_CORNERS: align_corners,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: output_shape,
        LazyAttr.OUT_DTYPE: dtype_,
    }
    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)


def zoom(
        img: torch.Tensor,
        factor: Union[Sequence[float], float],
        mode: Optional[Union[InterpolateMode, str]] = InterpolateMode.BILINEAR,
        padding_mode: Optional[Union[NumpyPadMode, GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        align_corners: Optional[bool] = False,
        keep_size: Optional[bool] = True,
        dtype: Optional[Union[DtypeLike, torch.dtype]] = None,
        shape_override: Optional[Sequence[int]] = None,
        dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
        lazy_evaluation: Optional[bool] = True
):
    """
    Args:
        img: channel first array, must have shape: (num_channels, H[, W, ..., ]).
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode. Defaults to ``self.mode``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'. Defaults to ``self.align_corners``.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

    Raises:
        ValueError: When ``self.spatial_size`` length is less than ``img`` spatial dimensions.

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    zoom_factors = ensure_tuple_rep(factor, input_ndim)
    # zoom_factors = ensure_tuple(factor)
    zoom_factors = [1 / f for f in zoom_factors]
    shape_zoom_factors = [1 / z for z in zoom_factors]

    mode_ = look_up_option(mode, GridSampleMode)
    padding_mode_ = 'border' if padding_mode == 'edge' else padding_mode
    padding_mode_ = look_up_option(padding_mode_, GridSamplePadMode)
    dtype_ = get_equivalent_dtype(dtype or img_.dtype, torch.Tensor)

    transform = create_scale(input_ndim, zoom_factors)
    shape_transform = create_scale(input_ndim, shape_zoom_factors)

    output_shape = input_shape if keep_size is True else transform_shape(input_shape,
                                                                         shape_transform)

    if align_corners is True:
        transform_ = apply_align_corners(transform, output_shape[1:],
                                         lambda scale_factors: create_scale(input_ndim, scale_factors))
        # TODO: confirm whether a second transform shape is required or not
        output_shape = transform_shape(output_shape, transform)
    else:
        transform_ = transform


    metadata = {
        "op": "zoom",
        "factor": zoom_factors,
        LazyAttr.INTERP_MODE: mode_,
        LazyAttr.PADDING_MODE: padding_mode_,
        LazyAttr.ALIGN_CORNERS: align_corners,
        "keep_size": keep_size,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: output_shape,
        LazyAttr.OUT_DTYPE: dtype_,
    }

    return lazily_apply_op(img_, MetaMatrix(transform_, metadata), lazy_evaluation)


def rotate90(
        img: torch.Tensor,
        k: Optional[int] = 1,
        spatial_axes: Optional[Tuple[int, int]] = (0, 1),
        shape_override: Optional[Sequence[int]] = None,
        dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
        lazy_evaluation: Optional[bool] = True
):
    """
    TODO: confirm the behaviour of rotate 90 when rotating non-square/non-cubic datasets"

    Args:
        img:
        k:
        spatial_axes:
        shape_override:
        lazy_evaluation:

    Returns:

    """
    if len(spatial_axes) != 2:
        raise ValueError("'spatial_axes' must be a tuple of two integers indicating")

    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    # if shape_override is set, it always wins
    input_shape = shape_override

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    transform = create_rotate_90(input_ndim, spatial_axes, k)

    # TODO: this could be calculated from the transform like the other functions do
    if k % 2 == 1:
        output_shape_order = [i for i in range(input_ndim)]
        for i in range(input_ndim):
            if i == spatial_axes[0]:
                output_shape_order[i] = spatial_axes[1]
            elif i == spatial_axes[1]:
                output_shape_order[i] = spatial_axes[0]
        output_shape = (input_shape[0],) + tuple(input_shape[output_shape_order[i] + 1] for i in range(input_ndim))
    else:
        output_shape = input_shape

    metadata = {
        "op": "rotate90",
        "k": k,
        "spatial_axes": spatial_axes,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: output_shape,
        LazyAttr.OUT_DTYPE: input_dtype,
    }
    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)


# TODO: Needs a second look
# def grid_distortion(
#         img: torch.Tensor,
#         num_cells: Union[Tuple[int], int],
#         distort_steps: Sequence[Sequence[float]],
#         mode: str = GridSampleMode.BILINEAR,
#         padding_mode: str = GridSamplePadMode.BORDER,
#         shape_override: Optional[Sequence[int]] = None,
#         lazy_evaluation: Optional[bool] = True
# ):
#     all_ranges = []
#     num_cells = ensure_tuple_rep(num_cells, len(img.shape) - 1)
#     for dim_idx, dim_size in enumerate(img.shape[1:]):
#         dim_distort_steps = distort_steps[dim_idx]
#         ranges = torch.zeros(dim_size, dtype=torch.float32)
#         cell_size = dim_size // num_cells[dim_idx]
#         prev = 0
#         for idx in range(num_cells[dim_idx] + 1):
#             start = int(idx * cell_size)
#             end = start + cell_size
#             if end > dim_size:
#                 end = dim_size
#                 cur = dim_size
#             else:
#                 cur = prev + cell_size * dim_distort_steps[idx]
#             prev = cur
#         ranges = range - (dim_size - 1.0) / 2.0
#         all_ranges.append()
#     coords = meshgrid_ij(*all_ranges)
#     grid = torch.stack([*coords, torch.ones_like(coords[0])])
#
#     metadata = {
#         "num_cells": num_cells,
#         "distort_steps": distort_steps,
#         "mode": mode,
#         "padding_mode": padding_mode
#     }
#
#     return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)


def elastic_3d(
        img: torch.Tensor,
        sigma: float,
        magnitude: float,
        offsets: torch.Tensor,
        spatial_size: Optional[Union[Tuple[int, int, int], int]] = None,
        mode: str = GridSampleMode.BILINEAR,
        padding_mode: str = GridSamplePadMode.REFLECTION,
        device: Optional[torch.device] = None,
        shape_override: Optional[Tuple[float]] = None,
        dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
        lazy_evaluation: Optional[bool] = True
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    sp_size = fall_back_tuple(spatial_size, img.shape[1:])
    device_ = img.device if isinstance(img, torch.Tensor) else device
    grid = create_grid(spatial_size=sp_size, device=device_, backend="torch")
    gaussian = GaussianFilter(3, sigma, 3.0).to(device=device_)
    grid[:3] += gaussian(offsets)[0] * magnitude

    metadata = {
        "op": "elastic_3d",
        "sigma": sigma,
        "magnitude": magnitude,
        "offsets": offsets,
        LazyAttr.SHAPE: input_shape
    }
    if spatial_size is not None:
        metadata["spatial_size"] = spatial_size
    if mode is not None:
        metadata[LazyAttr.INTERP_MODE] = mode
    if padding_mode is not None:
        metadata[LazyAttr.PADDING_MODE] = padding_mode

    return lazily_apply_op(img_, MetaMatrix(grid, metadata), lazy_evaluation)


def translate(
        img: torch.Tensor,
        translation: Sequence[float],
        mode: Optional[Union[GridSampleMode, str]] = GridSampleMode.BILINEAR,
        padding_mode: Optional[Union[GridSamplePadMode, str]] = NumpyPadMode.EDGE,
        dtype: Union[DtypeLike, torch.dtype] = np.float32,
        shape_override: Optional[Sequence[int]] = None,
        dtype_override: Optional[Union[DtypeLike, torch.dtype]] = None,
        lazy_evaluation: Optional[bool] = True
):
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    dtype_ = img_.dtype if dtype is None else dtype
    input_ndim = len(input_shape) - 1
    if len(translation) != input_ndim:
        raise ValueError(f"'translate' length {len(translation)} must be equal to 'img' "
                         f"spatial dimensions of {input_ndim}")

    transform = compatible_translate(img, translation)

    metadata = {
        "op": "translation",
        "translation": translation,
        LazyAttr.INTERP_MODE: mode,
        LazyAttr.PADDING_MODE: padding_mode,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: shape_override,
        LazyAttr.OUT_DTYPE: dtype_,
    }

    return lazily_apply_op(img_, MetaMatrix(transform, metadata), lazy_evaluation)
