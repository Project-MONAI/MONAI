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
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import AFFINE_TOL, compute_shape_offset, to_affine_nd
from monai.networks.layers import AffineTransform
from monai.networks.utils import normalize_transform
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import create_scale
from monai.transforms.utils_pytorch_numpy_unification import allclose
from monai.utils import convert_to_dst_type, convert_to_tensor, ensure_tuple, fall_back_tuple, pytorch_after
from monai.utils.enums import TraceKeys

__all__ = ["spatial_resample"]


def spatial_resample(
    img, dst_affine, spatial_size, mode, padding_mode, align_corners, dtype, transform_info
) -> torch.Tensor:
    original_spatial_shape = img.shape[1:]

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

    if (
        allclose(src_affine_, dst_affine, atol=AFFINE_TOL)
        and allclose(spatial_size, in_spatial_size)
        or spatial_rank == 1
    ):
        # no significant change, return original image
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        if get_track_meta():
            img.affine = dst_affine
        return TraceableTransform.track_transform(
            img,
            extra_info={
                "dtype": str(dtype_)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
                "mode": mode.value if isinstance(mode, Enum) else mode,
                "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                "src_affine": src_affine_,
            },
            orig_size=original_spatial_shape,
            transform_info=transform_info,
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
        return TraceableTransform.track_pending_transform(
            img,
            lazy_shape=spatial_size,
            lazy_affine=xform,
            orig_size=original_spatial_shape,
            extra_info={
                "dtype": str(dtype_)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
                "mode": mode.value if isinstance(mode, Enum) else mode,
                "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                "src_affine": src_affine_,
            },
            transform_info=transform_info,
        )

    # no resampling if it's identity transform
    if allclose(xform, torch.eye(len(xform)), atol=AFFINE_TOL) and allclose(spatial_size, in_spatial_size):
        img = convert_to_tensor(img, track_meta=get_track_meta(), dtype=torch.float32)
        if get_track_meta():
            img.affine = dst_affine
        return TraceableTransform.track_transform(
            img,
            extra_info={
                "dtype": str(dtype_)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
                "mode": mode.value if isinstance(mode, Enum) else mode,
                "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
                "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
                "src_affine": src_affine_,
            },
            orig_size=original_spatial_shape,
            transform_info=transform_info,
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
    return TraceableTransform.track_transform(
        img,
        extra_info={
            "dtype": str(dtype_)[6:],  # dtype as string; remove "torch": torch.float32 -> float32
            "mode": mode.value if isinstance(mode, Enum) else mode,
            "padding_mode": padding_mode.value if isinstance(padding_mode, Enum) else padding_mode,
            "align_corners": align_corners if align_corners is not None else TraceKeys.NONE,
            "src_affine": src_affine_,
        },
        orig_size=original_spatial_shape,
        transform_info=transform_info,
    )
