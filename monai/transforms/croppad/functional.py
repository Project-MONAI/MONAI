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
from monai.utils import TraceKeys, convert_to_tensor, ensure_tuple

__all__ = ["crop_func"]


def crop_func(img, slices, transform_info):
    """

    """
    img_size = img.peek_pending_shape() if isinstance(img, MetaTensor) else img.shape[1:]
    spatial_rank = img.peek_pending_rank() if isinstance(img, MetaTensor) else 3
    cropped = np.asarray([[s.indices(o)[0], o - s.indices(o)[1]] for s, o in zip(slices[1:], img_size)])
    extra_info = {"cropped": cropped.flatten().tolist()}
    to_shift = [s.start if s.start is not None else 0 for s in ensure_tuple(slices)[1:]]
    shape = [s.indices(o)[1] - s.indices(o)[0] for s, o in zip(slices[1:], img_size)]
    meta_info = TraceableTransform.track_transform_meta(
        img,
        sp_size=shape,
        affine=create_translate(spatial_rank, to_shift),
        extra_info=extra_info,
        orig_size=img_size,
        transform_info=transform_info,
        lazy_evaluation=transform_info.get(TraceKeys.LAZY_EVALUATION, False),
    )
    out = convert_to_tensor(img.as_tensor() if isinstance(img, MetaTensor) else img, track_meta=get_track_meta())
    if transform_info.get(TraceKeys.LAZY_EVALUATION, False):
        return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else meta_info
    out = out[slices]
    return out.copy_meta_from(meta_info) if isinstance(out, MetaTensor) else out
