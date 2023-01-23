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
    convert_to_dst_type,
    convert_to_numpy,
    convert_to_tensor,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    optional_import,
    pytorch_after,
)
from monai.utils.enums import TraceKeys
from monai.utils.type_conversion import convert_data_type

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = ["spatial_resample", "orientation", "flip", "resize", "rotate", "zoom", "rotate90", "affine_func"]


def spatial_resample(
    img, dst_affine, spatial_size, mode, padding_mode, align_corners, dtype, transform_info
) -> torch.Tensor:
    original_spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    src_affine_: torch.Tensor = img.affine if isinstance(img, MetaTensor) else torch.eye(4)
    img = convert_to_tensor(data=img, track_meta=get_track_meta(), dtype=dtype)
    spatial_rank = min(len(img.shape) - 1, src_affine_.shape[0] - 1, 3)
    if (not isinstance(spatial_size, int) or spatial_size != -1) and spatial_size is not None:
        spatial_rank = min(len(ensure_tuple(spatial_size)), 3)  # infer spatial rank based on spatial_size
    src_affine_ = to_affine_nd(spatial_rank, src_affine_).to(dtype)
    dst_affine = to_affine_nd(spatial_rank, dst_affine) if dst_affine is not None else src_affine_
    dst_affine = convert_to_dst_type(dst_affine, src_affine_)[0]
    if not isinstance(dst_affine, torch.Tensor):
        raise ValueError(f"dst_affine should be a torch.Tensor, got {type(dst_affine)}")

    in_spatial_size = torch.tensor(img.shape[1 : spatial_rank + 1])
    if isinstance(spatial_size, int) and (spatial_size == -1):  # using the input spatial size
        spatial_size = in_spatial_size
    elif spatial_size is None and spatial_rank > 1:  # auto spatial size
        spatial_size, _ = compute_shape_offset(in_spatial_size, src_affine_, dst_affine)  # type: ignore
    spatial_size = torch.tensor(fall_back_tuple(ensure_tuple(spatial_size)[:spatial_rank], in_spatial_size))
    dtype_ = img.dtype
    extra_info = {
        "dtype": str(dtype_)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
        "mode": mode.value if isinstance(mode, Enum) else mode,
        "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "src_affine": src_affine_,
    }

    if (
        allclose(src_affine_, dst_affine, atol=AFFINE_TOL)
        and allclose(spatial_size, in_spatial_size)
        or spatial_rank == 1
    ):
        # no significant change, return original image
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        if get_track_meta():
            img.affine = dst_affine
        return TraceableTransform.track_transform(  # type: ignore
            img, extra_info=extra_info, orig_size=original_spatial_shape, transform_info=transform_info
        )
    try:
        _s = convert_to_tensor(src_affine_, track_meta=False, device=torch.device("cpu"))
        _d = convert_to_tensor(dst_affine, track_meta=False, device=torch.device("cpu"))
        xform = torch.linalg.solve(_s, _d) if pytorch_after(1, 8, 0) else torch.solve(_d, _s).solution  # type: ignore
    except (np.linalg.LinAlgError, RuntimeError) as e:
        raise ValueError("src affine is not invertible.") from e
    xform = to_affine_nd(spatial_rank, xform).to(device=img.device, dtype=dtype)
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        return TraceableTransform.track_pending_transform(  # type: ignore
            img,
            lazy_shape=spatial_size,
            lazy_affine=xform,
            orig_size=original_spatial_shape,
            extra_info=extra_info,
            transform_info=transform_info,
        )

    # no resampling if it's identity transform
    if allclose(xform, torch.eye(len(xform)), atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size):
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        if get_track_meta():
            img.affine = dst_affine
        return TraceableTransform.track_transform(  # type: ignore
            img, extra_info=extra_info, orig_size=original_spatial_shape, transform_info=transform_info
        )

    in_spatial_size = in_spatial_size.tolist()  # type: ignore
    chns, additional_dims = img.shape[0], img.shape[spatial_rank + 1 :]  # beyond three spatial dims

    if additional_dims:
        xform_shape = [-1] + in_spatial_size
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
    if get_track_meta():
        img.affine = dst_affine
    return TraceableTransform.track_transform(  # type: ignore
        img, extra_info=extra_info, orig_size=original_spatial_shape, transform_info=transform_info
    )


def orientation(data_array, original_affine, spatial_ornt, transform_info):
    spatial_shape = data_array.peek_pending_shape() if isinstance(data_array, MetaTensor) else data_array.shape[1:]
    affine_x = nib.orientations.inv_ornt_aff(spatial_ornt, spatial_shape)
    data_array = convert_to_tensor(data_array, track_meta=get_track_meta())

    spatial_ornt[:, 0] += 1  # skip channel dim
    spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
    axes = [ax for ax, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
    full_transpose = np.arange(len(spatial_shape) + 1)  # channel-first array
    full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
    extra_info = {"original_affine": original_affine}
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        if not get_track_meta():
            return data_array
        shape_np = convert_to_numpy(data_array.peek_pending_shape(), wrap_sequence=True)
        shape_np = shape_np[[i - 1 for i in full_transpose if i != 0]]
        return TraceableTransform.track_pending_transform(
            data_array, lazy_shape=shape_np, lazy_affine=affine_x, extra_info=extra_info, transform_info=transform_info
        )
    if axes:
        data_array = torch.flip(data_array, dims=axes)
    if not np.all(full_transpose == np.arange(len(data_array.shape))):
        data_array = data_array.permute(full_transpose.tolist())

    if get_track_meta():
        new_affine = to_affine_nd(len(spatial_shape), original_affine) @ affine_x
        new_affine = to_affine_nd(original_affine, new_affine)
        new_affine, *_ = convert_data_type(new_affine, torch.Tensor, dtype=torch.float32, device=data_array.device)
        data_array.affine = new_affine
    return TraceableTransform.track_transform(data_array, extra_info=extra_info, transform_info=transform_info)


def flip(img, shape, sp_axes, transform_info):
    def update_meta(affine, shape, axes):
        # shape and axes include the channel dim
        mat = convert_to_dst_type(torch.eye(len(affine)), affine)[0]
        for axis in axes:
            sp = axis - 1
            mat[sp, sp], mat[sp, -1] = mat[sp, sp] * -1, shape[axis] - 1
        return mat

    extra_info = {"axes": sp_axes}  # track the spatial axes
    axes = monai.transforms.utils.map_spatial_axes(img.ndim, sp_axes)  # use the axes with channel dim
    _affine = (
        img.peek_pending_affine()
        if isinstance(img, MetaTensor)
        else torch.eye(4, device=torch.device("cpu"), dtype=torch.float64)
    )
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        if not get_track_meta():
            return img
        _affine = update_meta(_affine, shape, axes)
        return TraceableTransform.track_pending_transform(
            img, lazy_shape=shape[1:], lazy_affine=_affine, extra_info=extra_info, transform_info=transform_info
        )

    out = torch.flip(img, axes)
    if get_track_meta():
        out.affine @= update_meta(_affine, shape, axes)  # type: ignore
    return TraceableTransform.track_transform(out, extra_info=extra_info, transform_info=transform_info)


def resize(img, out_size, mode, align_corners, input_ndim, anti_aliasing, anti_aliasing_sigma, transform_info):
    img = convert_to_tensor(img, track_meta=get_track_meta())
    orig_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    extra_info = {
        "mode": mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "new_dim": len(orig_size) - input_ndim,
    }
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        if anti_aliasing:
            raise ValueError("anti-aliasing is not compatible with lazy evaluation.")
        if not get_track_meta():
            return img  # type: ignore
        affine = convert_to_tensor(img.peek_pending_affine(), track_meta=False)
        _affine = scale_affine(affine, orig_size, out_size)
        return TraceableTransform.track_pending_transform(
            img,
            lazy_shape=out_size,
            lazy_affine=_affine,
            orig_size=orig_size,
            extra_info=extra_info,
            transform_info=transform_info,
        )
    if tuple(convert_to_numpy(orig_size)) == out_size:  # spatial shape is already the desired
        if not get_track_meta():
            return img
        affine = convert_to_tensor(img.peek_pending_affine(), track_meta=False)
        img.affine @= scale_affine(affine, orig_size, out_size)
        return TraceableTransform.track_transform(
            img, orig_size=orig_size, extra_info=extra_info, transform_info=transform_info
        )
    img_ = convert_to_tensor(img, dtype=torch.float, track_meta=False)

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

    img = convert_to_tensor(img, track_meta=get_track_meta())
    resized = torch.nn.functional.interpolate(
        input=img_.unsqueeze(0), size=out_size, mode=mode, align_corners=align_corners
    )
    out, *_ = convert_to_dst_type(resized.squeeze(0), img)
    if not get_track_meta():
        return out
    affine = convert_to_tensor(out.peek_pending_affine(), track_meta=False)
    out.affine @= scale_affine(affine, orig_size, out_size)
    return TraceableTransform.track_transform(
        out, orig_size=orig_size, extra_info=extra_info, transform_info=transform_info
    )


def rotate(img, angle, output_shape, mode, padding_mode, align_corners, dtype, transform_info):
    im_shape = np.asarray(img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:])
    input_ndim = len(im_shape)
    if input_ndim not in (2, 3):
        raise ValueError(f"Unsupported image dimension: {input_ndim}, available options are [2, 3].")
    _angle = ensure_tuple_rep(angle, 1 if input_ndim == 2 else 3)
    transform = create_rotate(input_ndim, _angle)
    if output_shape is None:
        corners = np.asarray(np.meshgrid(*[(0, dim) for dim in im_shape], indexing="ij")).reshape((len(im_shape), -1))
        corners = transform[:-1, :-1] @ corners  # type: ignore
        output_shape = np.asarray(corners.ptp(axis=1) + 0.5, dtype=int)
    shift = create_translate(input_ndim, ((im_shape - 1) / 2).tolist())
    shift_1 = create_translate(input_ndim, (-(output_shape - 1) / 2).tolist())
    transform = shift @ transform @ shift_1

    img_t = img.to(dtype)
    transform_t, *_ = convert_to_dst_type(transform, img_t)
    extra_info = {
        "rot_mat": transform,
        "mode": mode,
        "padding_mode": padding_mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "dtype": str(dtype)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
    }

    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        if not get_track_meta():
            return img  # type: ignore
        affine = convert_to_tensor(img.peek_pending_affine(), track_meta=False)
        mat = to_affine_nd(len(affine) - 1, transform_t)
        _affine = convert_to_dst_type(mat, affine)[0]
        _shape = img.peek_pending_shape()
        return TraceableTransform.track_pending_transform(
            img,
            orig_size=_shape,
            lazy_affine=_affine,
            lazy_shape=output_shape,
            extra_info=extra_info,
            transform_info=transform_info,
        )
    xform = AffineTransform(
        normalized=False, mode=mode, padding_mode=padding_mode, align_corners=align_corners, reverse_indexing=True
    )
    output: torch.Tensor = xform(img_t.unsqueeze(0), transform_t, spatial_size=output_shape).float().squeeze(0)
    out, *_ = convert_to_dst_type(output, dst=img, dtype=output.dtype)
    if get_track_meta():
        affine = convert_to_tensor(img.peek_pending_affine(), track_meta=False)
        mat = to_affine_nd(len(affine) - 1, transform_t)
        out.affine @= convert_to_dst_type(mat, affine)[0]
    return TraceableTransform.track_transform(
        out, orig_size=img_t.shape[1:], extra_info=extra_info, transform_info=transform_info
    )


def zoom(img, scale_factor, output_size, mode, padding_mode, align_corners, transform_info):
    extra_info = {
        "mode": mode,
        "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
        "do_padcrop": False,
        "padcrop": {},
    }
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        if not get_track_meta():
            return img  # type: ignore
        _shape = img.peek_pending_shape()
        affine = convert_to_tensor(img.peek_pending_affine(), track_meta=False)
        _affine = scale_affine(affine, _shape, output_size)
        return TraceableTransform.track_pending_transform(
            img,
            orig_size=_shape,
            lazy_shape=output_size,
            lazy_affine=_affine,
            extra_info=extra_info,
            transform_info=transform_info,
        )
    img_t = img.to(torch.float32)
    zoomed: NdarrayOrTensor = torch.nn.functional.interpolate(
        recompute_scale_factor=True,
        input=img_t.unsqueeze(0),
        scale_factor=list(scale_factor),
        mode=mode,
        align_corners=align_corners,
    )
    zoomed = zoomed.squeeze(0)
    orig_size, z_size = img_t.shape, zoomed.shape
    out, *_ = convert_to_dst_type(zoomed, dst=img)
    if get_track_meta():
        affine = convert_to_tensor(out.peek_pending_affine(), track_meta=False)
        out.affine @= scale_affine(affine, orig_size[1:], z_size[1:])
    do_pad_crop = not np.allclose(output_size, z_size[1:])
    if do_pad_crop:
        _pad_crop = ResizeWithPadOrCrop(spatial_size=img_t.shape[1:], mode=padding_mode)
        out = _pad_crop(out)
    if get_track_meta() and do_pad_crop:
        extra_info["do_padcrop"] = True
        extra_info["padcrop"] = out.applied_operations.pop()  # TODO: using applied_operations?
    return TraceableTransform.track_transform(
        out, orig_size=orig_size[1:], extra_info=extra_info, transform_info=transform_info
    )


def rotate90(img, axes, k, transform_info):
    def update_meta(img, spatial_size, new_spatial_size, axes, k):
        affine = convert_data_type(img.peek_pending_affine(), torch.Tensor)[0]
        r, sp_r = len(affine) - 1, len(spatial_size)
        mat = to_affine_nd(r, create_translate(sp_r, [-float(d - 1) / 2 for d in new_spatial_size]))
        s = -1.0 if int(axes[0]) - int(axes[1]) in (-1, 2) else 1.0
        if sp_r == 2:
            rot90 = to_affine_nd(r, create_rotate(sp_r, [s * np.pi / 2]))
        else:
            idx = {1, 2, 3} - set(axes)
            angle: list[float] = [0, 0, 0]
            angle[idx.pop() - 1] = s * np.pi / 2
            rot90 = to_affine_nd(r, create_rotate(sp_r, angle))
        for _ in range(k):
            mat = rot90 @ mat
        mat = to_affine_nd(r, create_translate(sp_r, [float(d - 1) / 2 for d in spatial_size])) @ mat
        return convert_to_dst_type(mat, affine)[0]

    extra_info = {"axes": [d - 1 for d in axes], "k": k}
    ori_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        if not get_track_meta():
            return img  # type: ignore
        output_shape = list(img.peek_pending_shape())
        if k in (1, 3):
            a_0, a_1 = axes[0] - 1, axes[1] - 1
            output_shape[a_0], output_shape[a_1] = ori_shape[a_1], ori_shape[a_0]
        _affine = update_meta(img, ori_shape, output_shape, axes, k)
        return TraceableTransform.track_pending_transform(
            img, lazy_shape=output_shape, lazy_affine=_affine, extra_info=extra_info, transform_info=transform_info
        )
    out: NdarrayOrTensor = torch.rot90(img, k, axes)
    out = convert_to_dst_type(out, img)[0]
    if get_track_meta():
        out.affine @= update_meta(out, ori_shape, out.shape[1:], axes, k)  # type: ignore
    return TraceableTransform.track_transform(out, extra_info=extra_info, transform_info=transform_info)


def affine_func(img, affine, grid, resampler, sp_size, _mode, _padding_mode, do_resampling, image_only, transform_info):
    extra_info = {"affine": affine, "mode": _mode, "padding_mode": _padding_mode, "do_resampling": do_resampling}
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        if not get_track_meta():
            return img  # type: ignore
        orig_affine = convert_data_type(img.peek_pending_affine(), torch.Tensor)[0]
        _affine = monai.transforms.Affine.compute_w_affine(orig_affine, affine, img_size, sp_size)
        img = TraceableTransform.track_pending_transform(
            img,
            orig_size=img_size,
            lazy_shape=sp_size,
            lazy_affine=_affine,
            extra_info=extra_info,
            transform_info=transform_info,
        )
        return img if image_only else (img, affine)
    if do_resampling:
        out = resampler(img=img, grid=grid, mode=_mode, padding_mode=_padding_mode)
    else:
        out = convert_data_type(img, dtype=torch.float32, device=resampler.device)[0]

    out = convert_to_tensor(out, track_meta=get_track_meta())
    if not isinstance(out, MetaTensor):
        return out if image_only else (out, affine)
    if get_track_meta():
        out.meta = img.meta
        orig_affine = convert_data_type(out.peek_pending_affine(), torch.Tensor)[0]
        out.affine @= monai.transforms.Affine.compute_w_affine(orig_affine, affine, img_size, sp_size)
    out = TraceableTransform.track_transform(
        out, orig_size=img_size, extra_info=extra_info, transform_info=transform_info
    )
    return out if image_only else (out, affine)
