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
A collection of "functional" transforms for spatial operations.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import math
import warnings
from enum import Enum

import numpy as np
import torch

import monai
from monai.config import USE_COMPILED
from monai.config.type_definitions import DtypeLike, NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.transforms.croppad.array import ResizeWithPadOrCrop
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.inverse import TraceableTransform
from monai.transforms.lazy.functional import lazily_apply_op
from monai.transforms.utils import (
    apply_align_corners,
    create_flip,
    create_rotate,
    create_rotate_90,
    create_scale,
    create_translate,
    get_input_shape_and_dtype,
    resolves_modes,
    scale_affine,
    transform_shape
)
from monai.transforms.utils_pytorch_numpy_unification import allclose
from monai.utils import (
    LazyAttr,
    TraceKeys,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    ensure_tuple_size,
    fall_back_tuple,
    look_up_option,
    optional_import,
)
from monai.utils.enums import (
    GridSampleMode,
    GridSamplePadMode,
    InterpolateMode,
    NumpyPadMode,
)
from monai.utils.type_conversion import (
    get_equivalent_dtype,
)

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = ["spatial_resample", "orientation", "flip", "resize", "rotate", "zoom", "rotate90", "affine_func"]


def _maybe_new_metatensor(img, dtype=None, device=None):
    """create a metatensor with fresh metadata if track_meta is True otherwise convert img into a torch tensor"""
    return convert_to_tensor(
        img.as_tensor() if isinstance(img, MetaTensor) else img,
        dtype=dtype,
        device=device,
        track_meta=get_track_meta(),
        wrap_sequence=True,
    )


def spatial_resample(
    img, dst_affine, spatial_size, mode, padding_mode, align_corners, dtype_pt, lazy, transform_info
) -> torch.Tensor:
    """
    Functional implementation of resampling the input image to the specified ``dst_affine`` matrix and ``spatial_size``.
    This function operates eagerly or lazily according to
    ``lazy`` (default ``False``).

    Args:
        img: data to be resampled, assuming `img` is channel-first.
        dst_affine: target affine matrix, if None, use the input affine matrix, effectively no resampling.
        spatial_size: output spatial size, if the component is ``-1``, use the corresponding input spatial size.
        mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
            Interpolation mode to calculate output values.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
            and the value represents the order of the spline interpolation.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When `mode` is an integer, using numpy/cupy backends, this argument accepts
            {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        align_corners: Geometrically, we consider the pixels of the input as squares rather than points.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype_pt: data `dtype` for resampling computation.
        lazy: a flag that indicates whether the operation should be performed lazily or not
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    original_spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    src_affine: torch.Tensor = img.peek_pending_affine() if isinstance(img, MetaTensor) else torch.eye(4)
    img = convert_to_tensor(data=img, track_meta=get_track_meta())
    # ensure spatial rank is <= 3
    spatial_rank = min(len(img.shape) - 1, src_affine.shape[0] - 1, 3)
    if (not isinstance(spatial_size, int) or spatial_size != -1) and spatial_size is not None:
        spatial_rank = min(len(ensure_tuple(spatial_size)), 3)  # infer spatial rank based on spatial_size
    src_affine = to_affine_nd(spatial_rank, src_affine).to(torch.float64)
    dst_affine = to_affine_nd(spatial_rank, dst_affine) if dst_affine is not None else src_affine
    dst_affine = convert_to_dst_type(dst_affine, src_affine)[0]
    if not isinstance(dst_affine, torch.Tensor):
        raise ValueError(f"dst_affine should be a torch.Tensor, got {type(dst_affine)}")

    in_spatial_size = torch.tensor(original_spatial_shape[:spatial_rank])
    if isinstance(spatial_size, int) and (spatial_size == -1):  # using the input spatial size
        spatial_size = in_spatial_size
    elif spatial_size is None and spatial_rank > 1:  # auto spatial size
        spatial_size, _ = compute_shape_offset(in_spatial_size, src_affine, dst_affine)  # type: ignore
    spatial_size = torch.tensor(
        fall_back_tuple(ensure_tuple(spatial_size)[:spatial_rank], in_spatial_size, lambda x: x >= 0)
    )
    extra_info = {
        "dtype": str(dtype_pt)[6:],  # remove "torch": torch.float32 -> float32
        "mode": mode.value if isinstance(mode, Enum) else mode,
        "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "src_affine": src_affine,
    }
    try:
        _s = convert_to_numpy(src_affine)
        _d = convert_to_numpy(dst_affine)
        xform = np.eye(spatial_rank + 1) if spatial_rank < 2 else np.linalg.solve(_s, _d)
    except (np.linalg.LinAlgError, RuntimeError) as e:
        raise ValueError(f"src affine is not invertible {_s}, {_d}.") from e
    xform = convert_to_tensor(to_affine_nd(spatial_rank, xform)).to(device=img.device, dtype=torch.float64)
    affine_unchanged = (
        allclose(src_affine, dst_affine, atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size)
    ) or (allclose(xform, np.eye(len(xform)), atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size))
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=spatial_size,
        affine=None if affine_unchanged and not lazy else xform,
        extra_info=extra_info,
        orig_size=original_spatial_shape,
        transform_info=transform_info,
        lazy=lazy,
    )
    if lazy:
        out = _maybe_new_metatensor(img)
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    if affine_unchanged:
        # no significant change or lazy change, return original image
        out = _maybe_new_metatensor(img, dtype=torch.float32)
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore
    # drop current meta first
    img = img.as_tensor() if isinstance(img, MetaTensor) else img
    im_size = list(img.shape)
    chns, in_sp_size, additional_dims = im_size[0], im_size[1 : spatial_rank + 1], im_size[spatial_rank + 1 :]

    if additional_dims:
        xform_shape = [-1] + in_sp_size
        img = img.reshape(xform_shape)
    img = img.to(dtype_pt)
    if isinstance(mode, int) or USE_COMPILED:
        dst_xform = create_translate(spatial_rank, [float(d - 1) / 2 for d in spatial_size])
        xform = xform @ convert_to_dst_type(dst_xform, xform)[0]
        affine_xform = monai.transforms.Affine(
            affine=xform,
            spatial_size=spatial_size,
            normalized=True,
            image_only=True,
            dtype=dtype_pt,
            align_corners=align_corners,
        )
        with affine_xform.trace_transform(False):
            img = affine_xform(img, mode=mode, padding_mode=padding_mode)
    else:
        _, _m, _p, _ = resolves_modes(mode, padding_mode)
        affine_xform = AffineTransform(  # type: ignore
            normalized=False, mode=_m, padding_mode=_p, align_corners=align_corners, reverse_indexing=True
        )
        img = affine_xform(img.unsqueeze(0), theta=xform.to(img), spatial_size=spatial_size).squeeze(0)  # type: ignore
    if additional_dims:
        full_shape = (chns, *spatial_size, *additional_dims)
        img = img.reshape(full_shape)
    out = _maybe_new_metatensor(img, dtype=torch.float32)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore


def orientation(img, original_affine, spatial_ornt, lazy, transform_info) -> torch.Tensor:
    """
    Functional implementation of changing the input image's orientation into the specified based on `spatial_ornt`.
    This function operates eagerly or lazily according to
    ``lazy`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        original_affine: original affine of the input image.
        spatial_ornt: orientations of the spatial axes,
            see also https://nipy.org/nibabel/reference/nibabel.orientations.html
        lazy: a flag that indicates whether the operation should be performed lazily or not
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    xform = nib.orientations.inv_ornt_aff(spatial_ornt, spatial_shape)
    img = convert_to_tensor(img, track_meta=get_track_meta())

    spatial_ornt[:, 0] += 1  # skip channel dim
    spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
    axes = [ax for ax, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
    full_transpose = np.arange(len(spatial_shape) + 1)  # channel-first array
    full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
    extra_info = {"original_affine": original_affine}

    shape_np = convert_to_numpy(spatial_shape, wrap_sequence=True)
    shape_np = shape_np[[i - 1 for i in full_transpose if i > 0]]
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=shape_np,
        affine=xform,
        extra_info=extra_info,
        orig_size=spatial_shape,
        transform_info=transform_info,
        lazy=lazy,
    )
    out = _maybe_new_metatensor(img)
    if lazy:
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    if axes:
        out = torch.flip(out, dims=axes)
    if not np.all(full_transpose == np.arange(len(out.shape))):
        out = out.permute(full_transpose.tolist())
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore


def flip(
        img: torch.Tensor,
        spatial_axis: Sequence[int] | int,
        shape_override: Sequence | None = None,
        dtype_override: DtypeLike | torch.dtype | None = None,
        lazy: bool = False
):
    """
    Functional implementation of flip.
    This function operates eagerly or lazily according to
    ``lazy`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        sp_axes: spatial axes along which to flip over.
            If None, will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.
        lazy: a flag that indicates whether the operation should be performed lazily or not
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    spatial_axis_ = spatial_axis
    if spatial_axis_ is None:
        spatial_axis_ = tuple(i for i in range(len(input_shape[1:])))

    transform = create_flip(input_ndim, spatial_axis_)

    metadata = {
        "transform": transform,
        "op": "flip",
        "spatial_axis": spatial_axis_,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: input_shape,
        LazyAttr.OUT_DTYPE: input_dtype,
    }
    return lazily_apply_op(img_, metadata, lazy)

def resize(
        img: torch.Tensor,
        spatial_size: Sequence[int] | int,
        size_mode: str = "all",
        mode: InterpolateMode | str = InterpolateMode.AREA,
        align_corners: bool = False,
        anti_aliasing: bool = None,
        anti_aliasing_sigma: Sequence[float] | float | None = None,
        dtype: DtypeLike | torch.dtype | None = None,
        shape_override: Sequence[int] | None = None,
        dtype_override: DtypeLike | torch.dtype | None = None,
        lazy: bool = False
):
    """
    Functional implementation of resize.
    This function operates eagerly or lazily according to
    ``lazy`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        out_size: expected shape of spatial dimensions after resize operation.
        mode: {``"nearest"``, ``"nearest-exact"``, ``"linear"``,
            ``"bilinear"``, ``"bicubic"``, ``"trilinear"``, ``"area"``}
            The interpolation mode.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        align_corners: This only has an effect when mode is
            'linear', 'bilinear', 'bicubic' or 'trilinear'.
        dtype: data type for resampling computation. If None, use the data type of input data.
        input_ndim: number of spatial dimensions.
        anti_aliasing: whether to apply a Gaussian filter to smooth the image prior
            to downsampling. It is crucial to filter when downsampling
            the image to avoid aliasing artifacts. See also ``skimage.transform.resize``
        anti_aliasing_sigma: {float, tuple of floats}, optional
            Standard deviation for Gaussian filtering used when anti-aliasing.
        lazy: a flag that indicates whether the operation should be performed lazily or not
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

    shape_transform = create_scale(input_ndim, shape_zoom_factors)
    pixel_transform = create_scale(input_ndim, pixel_zoom_factors)

    output_shape = transform_shape(input_shape, shape_transform)

    metadata = {
        "transform": pixel_transform,
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
    return lazily_apply_op(img_, metadata, lazy)


def rotate(
        img: torch.Tensor,
        angle: Sequence[float] | float,
        keep_size: bool = True,
        mode: InterpolateMode | str = InterpolateMode.AREA,
        padding_mode: NumpyPadMode | GridSamplePadMode | str = NumpyPadMode.EDGE,
        align_corners: bool = False,
        dtype: DtypeLike | torch.dtype = None,
        shape_override: Sequence[int] | None = None,
        dtype_override: DtypeLike | torch.dtype = None,
        lazy: bool = False
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
        "transform": transform,
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
    return lazily_apply_op(img_, metadata, lazy)


def zoom(
        img: torch.Tensor,
        factor: Sequence[float] | float,
        mode: InterpolateMode | str = InterpolateMode.BILINEAR,
        padding_mode: NumpyPadMode | GridSamplePadMode | str = NumpyPadMode.EDGE,
        align_corners: bool = False,
        keep_size: bool = True,
        dtype: DtypeLike | torch.dtype | None = None,
        shape_override: Sequence[int] | None = None,
        dtype_override: DtypeLike | torch.dtype | None = None,
        lazy: bool = False
):
    """
    Functional implementation of zoom.
    This function operates eagerly or lazily according to
    ``lazy`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        scale_factor: The zoom factor along the spatial axes.
            If a float, zoom is the same for each spatial axis.
            If a sequence, zoom should contain one value for each spatial axis.
        keep_size: Whether keep original size (padding/slicing if needed).
        mode: {``"bilinear"``, ``"nearest"``}
            Interpolation mode to calculate output values.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        align_corners: See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        dtype: data type for resampling computation.
            If None, use the data type of input data. To be compatible with other modules,
            the output data type is always ``float32``.
        lazy: a flag that indicates whether the operation should be performed lazily or not
        transform_info: a dictionary with the relevant information pertaining to an applied transform.

    """

    img_ = convert_to_tensor(img, track_meta=get_track_meta())

    input_shape, input_dtype = get_input_shape_and_dtype(shape_override, dtype_override, img_)

    input_ndim = len(input_shape) - 1

    zoom_factors = ensure_tuple_rep(factor, input_ndim)
    # zoom_factors = ensure_tuple(factor)
    zoom_factors = [1 / f for f in zoom_factors]
    shape_zoom_factors = [1 / z for z in zoom_factors]

    # TODO: Remove this after consolidated resampling
    mode_ = 'bilinear' if mode == 'area' else mode
    mode_ = look_up_option(mode_, GridSampleMode)
    # TODO: Remove this after consolidated resampling
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
        "transform": transform_,
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

    return lazily_apply_op(img_, metadata, lazy)


def rotate90(
        img: torch.Tensor,
        k: int = 1,
        spatial_axes: Tuple[int, int] = (0, 1),
        shape_override: Sequence[int] | None = None,
        dtype_override: DtypeLike | torch.dtype | None = None,
        lazy: bool = False
):
    """
    Functional implementation of rotate90.
    This function operates eagerly or lazily according to
    ``lazy`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
            If axis is negative it counts from the last to the first axis.
        k: number of times to rotate by 90 degrees.
        lazy: a flag that indicates whether the operation should be performed lazily or not
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
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
        "transform": "transform",
        "op": "rotate90",
        "k": k,
        "spatial_axes": spatial_axes,
        LazyAttr.IN_SHAPE: input_shape,
        LazyAttr.IN_DTYPE: input_dtype,
        LazyAttr.OUT_SHAPE: output_shape,
        LazyAttr.OUT_DTYPE: input_dtype,
    }
    return lazily_apply_op(img_, metadata, lazy)


def affine_func(
    img, affine, grid, resampler, sp_size, mode, padding_mode, do_resampling, image_only, lazy, transform_info
):
    """
    Functional implementation of affine.
    This function operates eagerly or lazily according to
    ``lazy`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        affine: the affine transformation to be applied, it can be a 3x3 or 4x4 matrix. This should be defined
            for the voxel space spatial centers (``float(size - 1)/2``).
        grid: used in non-lazy mode to pre-compute the grid to do the resampling.
        resampler: the resampler function, see also: :py:class:`monai.transforms.Resample`.
        sp_size: output image spatial size.
        mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
            Interpolation mode to calculate output values.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
            and the value represents the order of the spline interpolation.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
            Padding mode for outside grid values.
            See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
            When `mode` is an integer, using numpy/cupy backends, this argument accepts
            {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
            See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
        do_resampling: whether to do the resampling, this is a flag for the use case of updating metadata but
            skipping the actual (potentially heavy) resampling operation.
        image_only: if True return only the image volume, otherwise return (image, affine).
        lazy: a flag that indicates whether the operation should be performed lazily or not
        transform_info: a dictionary with the relevant information pertaining to an applied transform.

    """

    # resampler should carry the align_corners and type info
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
    extra_info = {
        "affine": affine,
        "mode": mode,
        "padding_mode": padding_mode,
        "do_resampling": do_resampling,
        "align_corners": resampler.align_corners,
    }
    affine = monai.transforms.Affine.compute_w_affine(rank, affine, img_size, sp_size)
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=sp_size,
        affine=affine,
        extra_info=extra_info,
        orig_size=img_size,
        transform_info=transform_info,
        lazy=lazy,
    )
    if lazy:
        out = _maybe_new_metatensor(img)
        out = out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
        return out if image_only else (out, affine)
    if do_resampling:
        out = resampler(img=img, grid=grid, mode=mode, padding_mode=padding_mode)
        out = _maybe_new_metatensor(out)
    else:
        out = _maybe_new_metatensor(img, dtype=torch.float32, device=resampler.device)
    out = out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    return out if image_only else (out, affine)
