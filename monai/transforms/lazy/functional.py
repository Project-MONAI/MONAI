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

from typing import Optional, Union

import numpy as np
import torch

from monai.data.meta_tensor import MetaTensor
from monai.data.utils import to_affine_nd
from monai.transforms.meta_matrix import matmul
from monai.transforms.utility.functional import resample
from monai.utils import LazyAttr

__all__ = ["apply"]


def mat_from_pending(pending_item):
    if isinstance(pending_item, (torch.Tensor, np.ndarray)):
        return pending_item
    if isinstance(pending_item, dict):
        return pending_item[LazyAttr.AFFINE]
    return pending_item


def kwargs_from_pending(pending_item):
    if not isinstance(pending_item, dict):
        return {}
    ret = {
        LazyAttr.INTERP_MODE: pending_item.get(LazyAttr.INTERP_MODE, None),  # interpolation mode
        LazyAttr.PADDING_MODE: pending_item.get(LazyAttr.PADDING_MODE, None),  # padding mode
    }
    if LazyAttr.SHAPE in pending_item:
        ret[LazyAttr.SHAPE] = pending_item[LazyAttr.SHAPE]
    if LazyAttr.DTYPE in pending_item:
        ret[LazyAttr.DTYPE] = pending_item[LazyAttr.DTYPE]
    return ret


def is_compatible_kwargs(kwargs_1, kwargs_2):
    return True


def apply(data: Union[torch.Tensor, MetaTensor], pending: Optional[list] = None):
    """
    This method applies pending transforms to tensors.

    Args:
        data: A torch Tensor, monai MetaTensor
        pending: pending transforms. This must be set if data is a Tensor, but is optional if data is a MetaTensor.
    """
    if isinstance(data, MetaTensor) and pending is None:
        pending = data.pending_operations
    pending = [] if pending is None else pending

    if not pending:
        return data

    cumulative_xform = mat_from_pending(pending[0])
    cur_kwargs = kwargs_from_pending(pending[0])

    for p in pending[1:]:
        new_kwargs = kwargs_from_pending(p)
        if not is_compatible_kwargs(cur_kwargs, new_kwargs):
            # carry out an intermediate resample here due to incompatibility between arguments
            data = resample(data, cumulative_xform, cur_kwargs)
        next_matrix = mat_from_pending(p)
        cumulative_xform = matmul(cumulative_xform, next_matrix)
        cur_kwargs.update(new_kwargs)
    data = resample(data, cumulative_xform, cur_kwargs)
    if isinstance(data, MetaTensor):
        data.clear_pending_operations()
        data.affine = data.affine @ to_affine_nd(3, cumulative_xform)
        for p in pending:
            data.push_applied_operation(p)

    return data, pending
