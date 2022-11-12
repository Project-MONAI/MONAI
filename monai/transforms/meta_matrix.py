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


import torch

from monai.config import NdarrayOrTensor

__all__ = ["is_affine_shaped", "matmul"]


def is_affine_shaped(data):
    """Check if the data is a square matrix for the last two dimensions."""
    if not hasattr(data, "shape") or len(data.shape) < 2:
        return False
    return data.shape[-1] in (3, 4) and data.shape[-2] in (3, 4) and data.shape[-1] == data.shape[-2]


def matmul(left: NdarrayOrTensor, right: NdarrayOrTensor):
    if is_affine_shaped(left) and is_affine_shaped(right):
        return torch.matmul(left, right)
    raise NotImplementedError
