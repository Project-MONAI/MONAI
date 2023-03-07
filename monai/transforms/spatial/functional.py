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
A collection of "vanilla" transforms for spatial operations
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

from __future__ import annotations

import warnings
from enum import Enum

import numpy as np
import torch

import monai
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.networks.utils import normalize_transform
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import create_rotate, create_scale, create_translate, scale_affine
from monai.transforms.utils_pytorch_numpy_unification import allclose
from monai.utils import (
    TraceKeys,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    optional_import,
)

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = ["spatial_resample", "orientation", "flip", "rotate"]


def spatial_resample(
    img, dst_affine, spatial_size, mode, padding_mode, align_corners, dtype_pt, transform_info
) -> torch.Tensor:
    """
    Functional implementation of resampling the input image to the specified ``dst_affine`` matrix and ``spatial_size``.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

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
    lazy_evaluation = transform_info.get(TraceKeys.LAZY_EVALUATION, False)
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=spatial_size,
        affine=None if affine_unchanged and not lazy_evaluation else xform,
        extra_info=extra_info,
        orig_size=original_spatial_shape,
        transform_info=transform_info,
        lazy_evaluation=lazy_evaluation,
    )
    # drop current meta first since line 102 is a shallow copy
    img = img.as_tensor() if isinstance(img, MetaTensor) else img
    if affine_unchanged or lazy_evaluation:
        # no significant change or lazy change, return original image
        out = convert_to_tensor(img, track_meta=get_track_meta())
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    im_size = list(img.shape)
    chns, in_sp_size, additional_dims = im_size[0], im_size[1 : spatial_rank + 1], im_size[spatial_rank + 1 :]

    if additional_dims:
        xform_shape = [-1] + in_sp_size
        img = img.reshape(xform_shape)
    img = img.to(dtype_pt)
    if isinstance(mode, int):
        dst_xform_1 = normalize_transform(spatial_size, "cpu", xform.dtype, True, True)[0].numpy()  # to (-1, 1)
        if not align_corners:
            norm = create_scale(spatial_rank, [(max(d, 2) - 1) / d for d in spatial_size])
            dst_xform_1 = norm.astype(float) @ dst_xform_1  # type: ignore # scaling (num_step - 1) / num_step
        dst_xform_d = normalize_transform(spatial_size, "cpu", xform.dtype, align_corners, False)[0].numpy()
        xform @= convert_to_dst_type(np.linalg.solve(dst_xform_d, dst_xform_1), xform)[0]
        affine_xform = monai.transforms.Affine(
            affine=xform, spatial_size=spatial_size, normalized=True, image_only=True, dtype=dtype_pt
        )
        with affine_xform.trace_transform(False):
            img = affine_xform(img, mode=mode, padding_mode=padding_mode)
    else:
        affine_xform = AffineTransform(  # type: ignore
            normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
        )
        img = affine_xform(img.unsqueeze(0), theta=xform.to(img), spatial_size=spatial_size).squeeze(0)  # type: ignore
    if additional_dims:
        full_shape = (chns, *spatial_size, *additional_dims)
        img = img.reshape(full_shape)
    out = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out  # type: ignore


def orientation(img, original_affine, spatial_ornt, transform_info):
    """
    Functional implementation of changing the input image's orientation into the specified based on `spatial_ornt`.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        original_affine: original affine of the input image.
        spatial_ornt: orientation.
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
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    if axes:
        out = torch.flip(out, dims=axes)
    if not np.all(full_transpose == np.arange(len(out.shape))):
        out = out.permute(full_transpose.tolist())
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def flip(img, sp_axes, transform_info):
    """
    Functional implementation of flip.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        sp_axes: spatial axes along which to flip over.
            If None, will flip over all of the axes of the input array.
            If axis is negative it counts from the last to the first axis.
            If axis is a tuple of ints, flipping is performed on all of the axes
            specified in the tuple.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    sp_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    sp_size = convert_to_numpy(sp_size, wrap_sequence=True).tolist()
    extra_info = {"axes": sp_axes}  # track the spatial axes
    axes = monai.transforms.utils.map_spatial_axes(img.ndim, sp_axes)  # use the axes with channel dim
    rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
    # axes include the channel dim
    xform = torch.eye(int(rank) + 1, dtype=torch.double)
    for axis in axes:
        sp = axis - 1
        xform[sp, sp], xform[sp, -1] = xform[sp, sp] * -1, sp_size[sp] - 1
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=sp_size,
        affine=xform,
        extra_info=extra_info,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    out = torch.flip(out, axes)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def resize(img, out_size, mode, align_corners, dtype, input_ndim, anti_aliasing, anti_aliasing_sigma, transform_info):
    """
    Functional implementation of resize.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

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
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    img = convert_to_tensor(img, track_meta=get_track_meta())
    orig_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    extra_info = {
        "mode": mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "dtype": str(dtype)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
        "new_dim": len(orig_size) - input_ndim,
    }
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=out_size,
        affine=scale_affine(orig_size, out_size),
        extra_info=extra_info,
        orig_size=orig_size,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False) or tuple(convert_to_numpy(orig_size)) == out_size:
        if anti_aliasing and transform_info.get(TraceKeys.LAZY_EVALUATION, False):
            warnings.warn("anti-aliasing is not compatible with lazy evaluation.")
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    img_ = convert_to_tensor(out, dtype=dtype, track_meta=False)  # convert to a regular tensor
    if anti_aliasing and any(x < y for x, y in zip(out_size, img_.shape[1:])):
        factors = torch.div(torch.Tensor(list(img_.shape[1:])), torch.Tensor(out_size))
        if anti_aliasing_sigma is None:
            # if sigma is not given, use the default sigma in skimage.transform.resize
            anti_aliasing_sigma = torch.maximum(torch.zeros(factors.shape), (factors - 1) / 2).tolist()
        else:
            # if sigma is given, use the given value for downsampling axis
            anti_aliasing_sigma = list(ensure_tuple_rep(anti_aliasing_sigma, len(out_size)))
            for axis in range(len(out_size)):
                anti_aliasing_sigma[axis] = anti_aliasing_sigma[axis] * int(factors[axis] > 1)
        anti_aliasing_filter = GaussianSmooth(sigma=anti_aliasing_sigma)
        img_ = convert_to_tensor(anti_aliasing_filter(img_), track_meta=False)
    resized = torch.nn.functional.interpolate(
        input=img_.unsqueeze(0), size=out_size, mode=mode, align_corners=align_corners
    )
    out, *_ = convert_to_dst_type(resized.squeeze(0), out, dtype=torch.float32)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def rotate(img, angle, output_shape, mode, padding_mode, align_corners, dtype, transform_info):
    """
    Functional implementation of rotate.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        angle: Rotation angle(s) in radians. should a float for 2D, three floats for 3D.
        output_shape: output shape of the rotated data.
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
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    im_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    input_ndim = len(im_shape)
    if input_ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimension: {input_ndim}, available options are [2, 3].")
    _angle = ensure_tuple_rep(angle, 1 if input_ndim == 2 else 3)
    transform = create_rotate(input_ndim, _angle)
    if output_shape is None:
        corners = np.asarray(np.meshgrid(*[(0, dim) for dim in im_shape], indexing="ij")).reshape((len(im_shape), -1))
        corners = transform[:-1, :-1] @ corners  # type: ignore
        output_shape = np.asarray(corners.ptp(axis=1) + 0.5, dtype=int)
    shift = create_translate(input_ndim, ((np.array(im_shape) - 1) / 2).tolist())
    shift_1 = create_translate(input_ndim, (-(np.asarray(output_shape, dtype=int) - 1) / 2).tolist())
    transform = shift @ transform @ shift_1
    extra_info = {
        "rot_mat": transform,
        "mode": mode,
        "padding_mode": padding_mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "dtype": str(dtype)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
    }
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=output_shape,
        affine=transform,
        extra_info=extra_info,
        orig_size=im_shape,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    xform = AffineTransform(
        normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
    )
    img_t = out.to(dtype)
    transform_t, *_ = convert_to_dst_type(transform, img_t)
    output: torch.Tensor = xform(img_t.unsqueeze(0), transform_t, spatial_size=tuple(int(i) for i in output_shape))
    output = output.float().squeeze(0)
    out, *_ = convert_to_dst_type(output, dst=out, dtype=torch.float32)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def rotate90(img, axes, k, transform_info):
    """
    Functional implementation of rotate90.
    This function operates eagerly or lazily according to
    ``transform_info[TraceKeys.LAZY_EVALUATION]`` (default ``False``).

    Args:
        img: data to be changed, assuming `img` is channel-first.
        axes: 2 int numbers, defines the plane to rotate with 2 spatial axes.
                Default: (0, 1), this is the first two axis in spatial dimensions.
                If axis is negative it counts from the last to the first axis.
        k: number of times to rotate by 90 degrees.
        transform_info: a dictionary with the relevant information pertaining to an applied transform.
    """
    extra_info = {"axes": [d - 1 for d in axes], "k": k}
    ori_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    sp_shape = list(ori_shape)
    if k in (1, 3):
        a_0, a_1 = axes[0] - 1, axes[1] - 1
        sp_shape[a_0], sp_shape[a_1] = ori_shape[a_1], ori_shape[a_0]
    rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
    r, sp_r = int(rank), len(ori_shape)
    xform = to_affine_nd(r, create_translate(sp_r, [-float(d - 1) / 2 for d in sp_shape]))
    s = -1.0 if int(axes[0]) - int(axes[1]) in (-1, 2) else 1.0
    if sp_r == 2:
        rot90 = to_affine_nd(r, create_rotate(sp_r, [s * np.pi / 2]))
    else:
        idx = {1, 2, 3} - set(axes)
        angle: list[float] = [0, 0, 0]
        angle[idx.pop() - 1] = s * np.pi / 2
        rot90 = to_affine_nd(r, create_rotate(sp_r, angle))
    for _ in range(k):
        xform = rot90 @ xform
    xform = to_affine_nd(r, create_translate(sp_r, [float(d - 1) / 2 for d in ori_shape])) @ xform
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=sp_shape,
        affine=xform,
        extra_info=extra_info,
        orig_size=ori_shape,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    out = torch.rot90(out, k, axes)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
