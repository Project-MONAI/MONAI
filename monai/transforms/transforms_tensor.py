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
from .compose import Transform
from monai.utils.misc import ensure_tuple


class Flip(Transform):
    """Reverses the order of elements along the given spatial axis. Preserves shape.
    Uses ``torch.flip`` in practice. See torch.flip for additional details.
    https://pytorch.org/docs/stable/torch.html#torch.flip

    Args:
        spatial_axis (tuple or list of ints): spatial axes along which to flip over.
    """

    def __init__(self, spatial_axis):
        self.axis = [i + 1 for i in ensure_tuple(spatial_axis)]

    def __call__(self, img):
        """
        Args:
            img (Tensor): channel first array, must have shape: (num_channels, H[, W, ..., ]),
        """
        return torch.flip(img, self.axis)
