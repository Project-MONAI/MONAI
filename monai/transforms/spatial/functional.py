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
)

nib, has_nib = optional_import("nibabel")
cupy, _ = optional_import("cupy")
cupy_ndi, _ = optional_import("cupyx.scipy.ndimage")
np_ndi, _ = optional_import("scipy.ndimage")

__all__ = ["spatial_resample", "orientation", "flip", "resize", "rotate", "zoom", "rotate90", "affine_func"]


def spatial_resample(
    img, dst_affine, spatial_size, mode, padding_mode, align_corners, dtype_pt, transform_info
) -> torch.Tensor:
    original_spatial_shape = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    src_affine: torch.Tensor = img.peek_pending_affine() if isinstance(img, MetaTensor) else torch.eye(4)
    img = convert_to_tensor(data=img, track_meta=get_track_meta())
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
    spatial_size = torch.tensor(fall_back_tuple(ensure_tuple(spatial_size)[:spatial_rank], in_spatial_size))
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
    img = img.as_tensor() if isinstance(img, MetaTensor) else img
    if affine_unchanged or lazy_evaluation:
        # no significant change or lazy change, return original image
        out = convert_to_tensor(img, track_meta=get_track_meta())
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info  # type: ignore
    im_size = torch.tensor(img.shape).tolist()
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
