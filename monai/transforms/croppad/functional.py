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
from monai.transforms.utils import create_translate
from monai.utils import TraceKeys, convert_to_tensor

__all__ = ["pad_func"]


def pad_func(img, to_pad_, mode, kwargs, transform_info):
    extra_info = {"padded": to_pad_}
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    spatial_rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else 3
    do_pad = np.asarray(to_pad_).any()
    if do_pad:
        to_pad_ = list(to_pad_)
        if len(to_pad_) < len(img.shape):
            to_pad_ = list(to_pad_) + [(0, 0)] * (len(img.shape) - len(to_pad_))
        to_shift = [-s[0] for s in to_pad_[1:]]  # skipping the channel pad
        xform = create_translate(spatial_rank, to_shift)
        shape = [d + s + e for d, (s, e) in zip(img_size, to_pad_[1:])]
    else:
        shape = img_size
        xform = torch.eye(int(spatial_rank) + 1, device=torch.device("cpu"), dtype=torch.float64)
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=shape,
        affine=xform,
        extra_info=extra_info,
        orig_size=img_size,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    out = monai.transforms.Pad.pad_nd(out, to_pad_, mode, **kwargs) if do_pad else out
    out = convert_to_tensor(out, track_meta=get_track_meta())
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
