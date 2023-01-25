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

import numpy as np
import torch

from monai.data.meta_tensor import MetaTensor
from monai.transforms.lazy.utils import (
    affine_from_pending,
    combine_transforms,
    is_compatible_apply_kwargs,
    kwargs_from_pending,
    resample,
)
from monai.utils.enums import LazyAttr

__all__ = ["apply_transforms"]


def apply_transforms(
    data: torch.Tensor | MetaTensor, pending: list | None = None, mode=None, padding_mode=None, dtype=np.float64
):
    """
    This method applies pending transforms to `data` tensors.
    TODO: docstring mode/padding mode overriding

    Args:
        data: A torch Tensor or a monai MetaTensor.
        pending: pending transforms. This must be set if data is a Tensor, but is optional if data is a MetaTensor.
    """
    if isinstance(data, MetaTensor) and pending is None:
        pending = data.pending_operations.copy()
        data.clear_pending_operations()
    pending = [] if pending is None else pending

    if not pending:
        return data, []
    cumulative_xform = affine_from_pending(pending[0])
    cur_kwargs = kwargs_from_pending(pending[0])
    overriding = {}
    if mode is not None:
        overriding[LazyAttr.INTERP_MODE] = mode
    if padding_mode is not None:
        overriding[LazyAttr.PADDING_MODE] = padding_mode
    overriding[LazyAttr.DTYPE] = dtype if dtype is not None else data.dtype

    for p in pending[1:]:
        new_kwargs = kwargs_from_pending(p)
        if not is_compatible_apply_kwargs(cur_kwargs, new_kwargs):
            # carry out an intermediate resample here due to incompatibility between arguments
            _cur_kwargs = cur_kwargs.copy()
            _cur_kwargs.update(overriding)
            sp_size = _cur_kwargs.pop(LazyAttr.SHAPE, None)
            data = resample(data, cumulative_xform, sp_size, _cur_kwargs)
        next_matrix = affine_from_pending(p)
        cumulative_xform = combine_transforms(cumulative_xform, next_matrix)
        cur_kwargs.update(new_kwargs)
    cur_kwargs.update(overriding)
    sp_size = cur_kwargs.pop(LazyAttr.SHAPE, None)
    data = resample(data, cumulative_xform, sp_size, cur_kwargs)
    if isinstance(data, MetaTensor):
        for p in pending:
            data.push_applied_operation(p)
    return data, pending
