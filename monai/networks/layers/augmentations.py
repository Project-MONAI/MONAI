# Copyright 2020 MONAI Consortium
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
import torch.nn as nn
from monai.utils.misc import ensure_tuple


class Flipn(nn.Module):
    """Reverses the order of elements along the given spatial axis.

    Args:
        spatial_axis (tuple or list of ints): spatial axes along which to flip over.
    """

    def __init__(self, spatial_axis):
        super().__init__()
        self.axis = [i + 2 for i in ensure_tuple(spatial_axis)]

    def forward(self, x):
        return torch.flip(x, self.axis)
