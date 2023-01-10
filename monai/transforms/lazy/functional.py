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

import torch

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.lazy.utils import (
    affine_from_pending,
    combine_transforms,
    is_compatible_apply_kwargs,
    kwargs_from_pending,
    resample,
)

__all__ = ["apply_transforms"]


def apply_transforms(data: torch.Tensor | MetaTensor, pending: list | None = None):
    """
    This method applies pending transforms to `data` tensors.

    Args:
        data: A torch Tensor or a monai MetaTensor.
        pending: pending transforms. This must be set if data is a Tensor, but is optional if data is a MetaTensor.
    """
    if isinstance(data, MetaTensor) and pending is None:
        pending = data.pending_operations
    pending = [] if pending is None else pending

    if not pending:
        return data

    cumulative_xform = affine_from_pending(pending[0])
    cur_kwargs = kwargs_from_pending(pending[0])

    for p in pending[1:]:
        new_kwargs = kwargs_from_pending(p)
        if not is_compatible_apply_kwargs(cur_kwargs, new_kwargs):
            # carry out an intermediate resample here due to incompatibility between arguments
            data = resample(data, cumulative_xform, cur_kwargs)
        next_matrix = affine_from_pending(p)
        cumulative_xform = combine_transforms(cumulative_xform, next_matrix)
        cur_kwargs.update(new_kwargs)
    data = resample(data, cumulative_xform, cur_kwargs)
    if isinstance(data, MetaTensor):
        data.clear_pending_operations()
        data.affine = data.affine @ to_affine_nd(3, cumulative_xform)
        for p in pending:
            data.push_applied_operation(p)

    return data, pending
