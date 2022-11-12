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
from typing import Optional

import torch

import monai
from monai.config import NdarrayOrTensor
from monai.transforms.meta_matrix import is_affine_shaped
from monai.utils import LazyAttr

__all__ = ["resample"]


def resample(data: torch.Tensor, matrix: NdarrayOrTensor, kwargs: Optional[dict] = None):
    """
    This is a minimal implementation of resample that always uses Affine.
    """
    if not is_affine_shaped(matrix):
        raise NotImplementedError("calling dense grid resample API not implemented")
    kwargs = {} if kwargs is None else kwargs
    init_kwargs = {
        "spatial_size": kwargs.pop(LazyAttr.SHAPE, data.shape)[1:],
        "dtype": kwargs.pop(LazyAttr.DTYPE, data.dtype),
    }
    call_kwargs = {
        "mode": kwargs.pop(LazyAttr.INTERP_MODE, None),
        "padding_mode": kwargs.pop(LazyAttr.PADDING_MODE, None),
    }
    resampler = monai.transforms.Affine(affine=matrix, image_only=True, **init_kwargs)
    with resampler.trace_transform(False):  # don't track this transform in `data`
        return resampler(img=data, **call_kwargs)
