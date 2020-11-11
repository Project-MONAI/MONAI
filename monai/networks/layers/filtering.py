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
    def forward(ctx, data, spatial_sigma=5, color_sigma=0.5):
        return _C.bilateral_filter(data, spatial_sigma, color_sigma)


    
# @staticmethod
#     def forward(ctx, data, spatial_sigma=5, color_sigma=0.5):

#         data = data.to(dtype=torch.float64)

#         # make a simple Gaussian function taking the squared radius
#         gaussian = lambda r2, sigma: np.exp(-0.5*r2/sigma**2)

#         # define the window width to be the 3 time the spatial std. dev. to 
#         # be sure that most of the spatial kernel is actually captured
#         win_width = int(3 * spatial_sigma)

#         # initialize the results and sum of weights to very small values for
#         # numerical stability. not strictly necessary but helpful to avoid
#         # wild values with pathological choices of parameters
#         wgt_sum = torch.ones_like(data) * 1e-8
#         result  = data * 1e-8

#         # accumulate the result by circularly shifting the image across the
#         # window in the horizontal and vertical directions. within the inner
#         # loop, calculate the two weights and accumulate the weight sum and 
#         # the unnormalized result image
#         for shft_x in range(-win_width, win_width+1):
#             for shft_y in range(-win_width, win_width+1):

#                 # shift by the offsets
#                 offset_data = torch.roll(data, [shft_y, shft_x], [0, 1])

#                 # compute weights
#                 spatial_weight = gaussian(shft_x**2 + shft_y**2, spatial_sigma)
#                 value_weight = torch.exp(-0.5 * ((offset_data - data)**2) / (color_sigma**2))
#                 total_weight = spatial_weight * value_weight

#                 # accumulate the results
#                 result += offset_data * total_weight
#                 wgt_sum += total_weight

#         # normalize the result and return
#         return result / wgt_sum