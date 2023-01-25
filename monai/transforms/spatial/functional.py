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

import math
import warnings
from enum import Enum

import numpy as np
import torch

import monai
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.networks.utils import normalize_transform
from monai.transforms.croppad.array import ResizeWithPadOrCrop
from monai.transforms.intensity.array import GaussianSmooth
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import create_rotate, create_scale, create_translate, scale_affine
from monai.transforms.utils_pytorch_numpy_unification import allclose
from monai.utils import (
    TraceKeys,
    convert_data_type,
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    optional_import,
    pytorch_after,
)

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = ["spatial_resample", "orientation", "flip", "resize", "rotate", "zoom", "rotate90", "affine_func"]


def spatial_resample(
    img, dst_affine, spatial_size, mode, padding_mode, align_corners, dtype, transform_info
) -> torch.Tensor:
    original_spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    src_affine: torch.Tensor = img.peek_pending_affine() if isinstance(img, MetaTensor) else torch.eye(4)
    img = convert_to_tensor(data=img, track_meta=get_track_meta(), dtype=dtype)
    spatial_rank = min(len(img.shape) - 1, src_affine.shape[0] - 1, 3)
    if (not isinstance(spatial_size, int) or spatial_size != -1) and spatial_size is not None:
        spatial_rank = min(len(ensure_tuple(spatial_size)), 3)  # infer spatial rank based on spatial_size
    src_affine = to_affine_nd(spatial_rank, src_affine).to(dtype)
    dst_affine = to_affine_nd(spatial_rank, dst_affine) if dst_affine is not None else src_affine
    dst_affine = convert_to_dst_type(dst_affine, src_affine)[0]
    if not isinstance(dst_affine, torch.Tensor):
        raise ValueError(f"dst_affine should be a torch.Tensor, got {type(dst_affine)}")

    in_spatial_size = torch.tensor(original_spatial_shape[:spatial_rank])
    if isinstance(spatial_size, int) and (spatial_size == -1):  # using the input spatial size
        spatial_size = in_spatial_size
    elif spatial_size is None and spatial_rank > 1:  # auto spatial size
        spatial_size, _ = compute_shape_offset(in_spatial_size, src_affine, dst_affine)  # type: ignore
    spatial_size = torch.tensor(fall_back_tuple(ensure_tuple(spatial_size)[:spatial_rank], in_spatial_size))
    extra_info = {
        "dtype": str(img.dtype)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
        "mode": mode.value if isinstance(mode, Enum) else mode,
        "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "src_affine": src_affine,
    }
    try:
        _s = convert_to_tensor(src_affine, track_meta=False, device=torch.device("cpu"))
        _d = convert_to_tensor(dst_affine, track_meta=False, device=torch.device("cpu"))
        if spatial_rank < 2:
            xform = torch.eye(spatial_rank + 1, device=torch.device("cpu"))
        elif pytorch_after(1, 8, 0):
            xform = torch.linalg.solve(_s, _d)
        else:
            xform = torch.solve(_d, _s).solution  # type: ignore
    except (np.linalg.LinAlgError, RuntimeError) as e:
        raise ValueError("src affine is not invertible.") from e
    xform = to_affine_nd(spatial_rank, xform).to(device=img.device, dtype=dtype)
    affine_unchanged = (
        allclose(src_affine, dst_affine, atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size)
    ) or (allclose(xform, torch.eye(len(xform)), atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size))
    lazy_evaluation = transform_info.get(TraceKeys.LAZY_EVALUATION, False)
    meta_info = TraceableTransform.track_transform_tensor(
        img,
        sp_size=spatial_size,
        affine=None if affine_unchanged and not lazy_evaluation else xform,
        extra_info=extra_info,
        orig_size=original_spatial_shape,
        transform_info=transform_info,
        lazy_evaluation=lazy_evaluation,
    )
    if affine_unchanged or lazy_evaluation:
        # no significant change or lazy change, return original image
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)  # type: ignore
        return img.copy_meta_from(meta_info) if isinstance(img, MetaTensor) else img  # type: ignore
    im_size = torch.tensor(img.shape).tolist()
    chns, in_sp_size, additional_dims = im_size[0], im_size[1 : spatial_rank + 1], im_size[spatial_rank + 1 :]

    if additional_dims:
        xform_shape = [-1] + in_sp_size
        img = img.reshape(xform_shape)  # type: ignore
    if isinstance(mode, int):
        dst_xform_1 = normalize_transform(spatial_size, xform.device, xform.dtype, True, True)[0]  # to (-1, 1)
        if not align_corners:
            norm = create_scale(spatial_rank, [(max(d, 2) - 1) / d for d in spatial_size], xform.device, "torch")
            dst_xform_1 = norm.to(xform.dtype) @ dst_xform_1  # type: ignore  # scaling (num_step - 1) / num_step
        dst_xform_d = normalize_transform(spatial_size, xform.device, xform.dtype, align_corners, False)[0]
        xform = xform @ torch.inverse(dst_xform_d) @ dst_xform_1
        affine_xform = monai.transforms.Affine(
            affine=xform, spatial_size=spatial_size, normalized=True, image_only=True, dtype=dtype
        )
        with affine_xform.trace_transform(False):
            img = affine_xform(img, mode=mode, padding_mode=padding_mode)
    else:
        affine_xform = AffineTransform(
            normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
        )
        img = affine_xform(img.unsqueeze(0), theta=xform, spatial_size=spatial_size).squeeze(0)
    if additional_dims:
        full_shape = (chns, *spatial_size, *additional_dims)
        img = img.reshape(full_shape)
    img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
    return img.copy_meta_from(meta_info) if isinstance(img, MetaTensor) else img  # type: ignore


def orientation(img, original_affine, spatial_ornt, transform_info):
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
    meta_info = TraceableTransform.track_transform_tensor(
        img,
        sp_size=shape_np,
        affine=xform,
        extra_info=extra_info,
        orig_size=spatial_shape,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    if axes:
        out = torch.flip(out, dims=axes)
    if not np.all(full_transpose == np.arange(len(out.shape))):
        out = out.permute(full_transpose.tolist())
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def flip(img, shape, sp_axes, transform_info):
    extra_info = {"axes": sp_axes}  # track the spatial axes
    axes = monai.transforms.utils.map_spatial_axes(img.ndim, sp_axes)  # use the axes with channel dim
    rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
    # shape and axes include the channel dim
    xform = convert_to_dst_type(torch.eye(int(rank) + 1), rank)[0]
    for axis in axes:
        sp = axis - 1
        xform[sp, sp], xform[sp, -1] = xform[sp, sp] * -1, shape[axis] - 1
    meta_info = TraceableTransform.track_transform_tensor(
        img,
        sp_size=shape[1:],
        affine=xform,
        extra_info=extra_info,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    out = torch.flip(out, axes)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def resize(img, out_size, mode, align_corners, input_ndim, anti_aliasing, anti_aliasing_sigma, transform_info):
    img = convert_to_tensor(img, track_meta=get_track_meta())
    orig_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
    extra_info = {
        "mode": mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "new_dim": len(orig_size) - input_ndim,
    }
    meta_info = TraceableTransform.track_transform_tensor(
        img,
        sp_size=out_size,
        affine=convert_to_dst_type(scale_affine(rank, orig_size, out_size), rank)[0],
        extra_info=extra_info,
        orig_size=orig_size,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False) or tuple(convert_to_numpy(orig_size)) == out_size:
        if anti_aliasing:
            warnings.warn("anti-aliasing is not compatible with lazy evaluation.")
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    img_ = convert_to_tensor(out, dtype=torch.float, track_meta=False)  # convert to a regular tensor
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
    out, *_ = convert_to_dst_type(resized.squeeze(0), out)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def rotate(img, angle, output_shape, mode, padding_mode, align_corners, dtype, transform_info):
    im_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    input_ndim = len(im_shape)
    if input_ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimension: {input_ndim}, available options are [2, 3].")
    _angle = ensure_tuple_rep(angle, 1 if input_ndim == 2 else 3)
    transform = create_rotate(input_ndim, _angle)
    if output_shape is None:
        corners = np.asarray(np.meshgrid(*[(0, dim) for dim in im_shape], indexing="ij")).reshape((len(im_shape), -1))
        corners = transform[:-1, :-1] @ corners  # type: ignore
        output_shape = corners.ptp(axis=1) + 0.5
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
    meta_info = TraceableTransform.track_transform_tensor(
        img,
        sp_size=output_shape,
        affine=transform,
        extra_info=extra_info,
        orig_size=im_shape,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    xform = AffineTransform(
        normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
    )
    img_t = out.to(dtype)
    transform_t, *_ = convert_to_dst_type(transform, img_t)
    output: torch.Tensor = xform(img_t.unsqueeze(0), transform_t, spatial_size=tuple(int(i) for i in output_shape))
    output = output.float().squeeze(0)
    out, *_ = convert_to_dst_type(output, dst=out, dtype=output.dtype)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def zoom(img, scale_factor, keep_size, mode, padding_mode, align_corners, transform_info):
    im_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
    output_size = [
        int(math.floor(float(i) * z))
        for i, z in zip(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:], scale_factor)
    ]
    xform = convert_to_dst_type(scale_affine(rank, im_shape, output_size), rank)[0]
    extra_info = {
        "mode": mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "do_padcrop": False,
        "padcrop": {},
    }
    if keep_size:
        if transform_info.get(TraceKeys.LAZY_EVALUATION):
            raise NotImplementedError("keep_size=True is not supported for lazy evaluation.")
        output_size = [int(i) for i in img.shape[1:]]
    meta_info = TraceableTransform.track_transform_tensor(
        img,
        sp_size=output_size,
        affine=xform,
        extra_info=extra_info,
        orig_size=im_shape,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    img_t = out.to(torch.float32)
    zoomed: NdarrayOrTensor = torch.nn.functional.interpolate(
        recompute_scale_factor=True,
        input=img_t.unsqueeze(0),
        scale_factor=list(scale_factor),
        mode=mode,
        align_corners=align_corners,
    ).squeeze(0)
    out, *_ = convert_to_dst_type(zoomed, dst=out)
    if isinstance(out, MetaTensor):
        out = out.copy_meta_from(meta_info)
    do_pad_crop = not np.allclose(output_size, zoomed.shape[1:])
    if do_pad_crop:
        _pad_crop = ResizeWithPadOrCrop(spatial_size=img_t.shape[1:], mode=padding_mode)
        out = _pad_crop(out)
    if get_track_meta() and do_pad_crop:
        extra_info["do_padcrop"] = True
        extra_info["padcrop"] = out.applied_operations.pop()  # TODO: using applied_operations?
    return out


def rotate90(img, axes, k, transform_info):
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
    meta_info = TraceableTransform.track_transform_tensor(
        img,
        sp_size=sp_shape,
        affine=xform,
        extra_info=extra_info,
        orig_size=ori_shape,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    out = torch.rot90(out, k, axes)
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out


def affine_func(img, affine, grid, resampler, sp_size, mode, padding_mode, do_resampling, image_only, transform_info):
    extra_info = {"affine": affine, "mode": mode, "padding_mode": padding_mode, "do_resampling": do_resampling}
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else torch.tensor(3.0, dtype=torch.double)
    affine = convert_to_dst_type(monai.transforms.Affine.compute_w_affine(rank, affine, img_size, sp_size), rank)[0]
    meta_info = TraceableTransform.track_transform_tensor(
        img,
        sp_size=sp_size,
        affine=affine,
        extra_info=extra_info,
        orig_size=img_size,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        out = out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
        return out if image_only else (out, affine)
    if do_resampling:
        out = resampler(img=out, grid=grid, mode=mode, padding_mode=padding_mode)
    else:
        out = convert_data_type(out, dtype=torch.float32, device=resampler.device)[0]
    out = convert_to_tensor(out, track_meta=get_track_meta())
    out = out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
    return out if image_only else (out, affine)
