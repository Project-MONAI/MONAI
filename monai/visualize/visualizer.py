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


from typing import Callable

import torch
import torch.nn.functional as F

from monai.utils import InterpolateMode

__all__ = ["default_upsampler"]


def default_upsampler(spatial_size, align_corners=False) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    A linear interpolation method for upsampling the feature map.
    The output of this function is a callable `func`,
    such that `func(x)` returns an upsampled tensor.
    """

    def up(x):

        linear_mode = [InterpolateMode.LINEAR, InterpolateMode.BILINEAR, InterpolateMode.TRILINEAR]
        interp_mode = linear_mode[len(spatial_size) - 1]
        return F.interpolate(x, size=spatial_size, mode=str(interp_mode.value), align_corners=align_corners)

    return up
