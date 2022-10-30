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

import torch
from monai.transforms import Affine

from monai.config import NdarrayOrTensor
from monai.transforms.meta_matrix import Grid, Matrix


def resample(
        data: torch.Tensor,
        matrix: Union[NdarrayOrTensor, Matrix, Grid],
        kwargs: Optional[dict] = None
):
    """
    This is a minimal implementation of resample that always uses Affine.
    """
    if kwargs is not None:
        a = Affine(affine=matrix, **kwargs)
    else:
        a = Affine(affine=matrix)
    return a(img=data)
