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

import numpy as np
import torch
from monai.utils.module import optional_import

_C, _ = optional_import("monai._C")

class BilateralFilter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, data, spatial_sigma=5, color_sigma=0.5, fast_approx=False):
        return _C.bilateral_filter(data, spatial_sigma, color_sigma, fast_approx)
