# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from monai.config.type_definitions import NdarrayOrTensor

__all__ = [
    "moveaxis",
]


def moveaxis(x: NdarrayOrTensor, src: int, dst: int) -> NdarrayOrTensor:
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "moveaxis"):
            return torch.moveaxis(x, src, dst)
        # moveaxis only available in pytorch as of 1.8.0
        else:
            # get original indices, remove desired index and insert it in new position
            indices = list(range(x.ndim))
            indices.pop(src)
            indices.insert(dst, src)
            return x.permute(indices)
    elif isinstance(x, np.ndarray):
        return np.moveaxis(x, src, dst)
    raise RuntimeError()
