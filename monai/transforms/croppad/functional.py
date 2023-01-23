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

import numpy as np
import torch

import monai
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms.inverse import TraceableTransform
from monai.transforms.utils import convert_pad_mode, create_translate
from monai.utils import TraceKeys, convert_to_dst_type, ensure_tuple

__all__ = ["pad_func", "crop_func"]


def pad_func(img_t, to_pad_, mode_, kwargs_, transform_info):
    extra_info = {"padded": to_pad_}
    img_size = img_t.peek_pending_shape() if isinstance(img_t, MetaTensor) else img_t.shape[1:]
    _affine = (
        img_t.peek_pending_affine()
        if isinstance(img_t, MetaTensor)
        else torch.eye(4, device=torch.device("cpu"), dtype=torch.float64)
    )
    spatial_rank = max(len(_affine) - 1, 1)
    if np.asarray(to_pad_).any():
        to_pad_ = list(to_pad_)
        if len(to_pad_) < len(img_t.shape):
            to_pad_ = list(to_pad_) + [(0, 0)] * (len(img_t.shape) - len(to_pad_))
        if transform_info.get(TraceKeys.LAZY_EVALUATION):
            if not get_track_meta():
                return img_t
            to_shift = [-s[0] for s in to_pad_[1:]]  # skipping the channel pad
            _affine = convert_to_dst_type(create_translate(spatial_rank, to_shift), _affine)[0]
            _shape = [d + s + e for d, (s, e) in zip(img_size, to_pad_[1:])]
            return TraceableTransform.track_pending_transform(
                img_t,
                orig_size=img_size,
                lazy_affine=_affine,
                lazy_shape=_shape,
                extra_info=extra_info,
                transform_info=transform_info,
            )
        if mode_ in {"linear_ramp", "maximum", "mean", "median", "minimum", "symmetric", "empty"}:
            out = monai.transforms.Pad._np_pad(img_t, pad_width=to_pad_, mode=mode_, **kwargs_)
        else:
            mode_ = convert_pad_mode(dst=img_t, mode=mode_).value
            try:
                _pad = (
                    monai.transforms.Pad._pt_pad
                    if mode_ in {"reflect", "replicate"}
                    and img_t.dtype not in {torch.int16, torch.int64, torch.bool, torch.uint8}
                    else monai.transforms.Pad._np_pad
                )
                out = _pad(img_t, pad_width=to_pad_, mode=mode_, **kwargs_)
            except (ValueError, TypeError, RuntimeError) as err:
                if isinstance(err, NotImplementedError) or any(
                    k in str(err) for k in ("supported", "unexpected keyword", "implemented")
                ):
                    out = monai.transforms.Pad._np_pad(img_t, pad_width=to_pad_, mode=mode_, **kwargs_)
                else:
                    raise ValueError(f"{img_t.shape} {to_pad_} {mode_} {kwargs_} {img_t.dtype} {img_t.device}") from err
    else:
        out = img_t
    if get_track_meta():
        to_shift = [-s[0] for s in to_pad_[1:]]  # skipping the channel pad
        out.affine @= convert_to_dst_type(create_translate(spatial_rank, to_shift), _affine)[0]  # type: ignore
    return TraceableTransform.track_transform(
        out, orig_size=img_size, extra_info={"padded": to_pad_}, transform_info=transform_info
    )


def crop_func(img_t, slices, transform_info):
    img_size = img_t.peek_pending_shape() if isinstance(img_t, MetaTensor) else img_t.shape[1:]
    _affine = (
        img_t.peek_pending_affine()
        if isinstance(img_t, MetaTensor)
        else torch.eye(4, device=torch.device("cpu"), dtype=torch.float64)
    )
    spatial_rank = max(len(_affine) - 1, 1)
    cropped = np.asarray([[s.indices(o)[0], o - s.indices(o)[1]] for s, o in zip(slices[1:], img_size)])
    extra_info = {"cropped": cropped.flatten().tolist()}
    if transform_info.get(TraceKeys.LAZY_EVALUATION):
        if not get_track_meta():
            return img_t
        to_shift = [s.start if s.start is not None else 0 for s in ensure_tuple(slices)[1:]]
        _affine = convert_to_dst_type(create_translate(spatial_rank, to_shift), _affine)[0]
        _shape = [s.indices(o)[1] - s.indices(o)[0] for s, o in zip(slices[1:], img_size)]
        return TraceableTransform.track_pending_transform(
            img_t,
            orig_size=img_size,
            lazy_shape=_shape,
            lazy_affine=_affine,
            extra_info=extra_info,
            transform_info=transform_info,
        )
    img_t = img_t[slices]  # type: ignore
    if get_track_meta():
        to_shift = [s.start if s.start is not None else 0 for s in ensure_tuple(slices)[1:]]
        mat = create_translate(spatial_rank, to_shift)
        img_t.affine @= convert_to_dst_type(mat, _affine)[0]
    return TraceableTransform.track_transform(
        img_t, orig_size=img_size, extra_info=extra_info, transform_info=transform_info
    )
