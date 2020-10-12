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

import unittest

import numpy as np
import torch
import torch.nn as nn
from parameterized import parameterized

from monai.metrics import compute_occlusion_sensitivity
from monai.networks.nets import densenet121

# # keep background
# TEST_CASE_1 = [  # y (1, 1, 2, 2), y_pred (1, 1, 2, 2), expected out (1, 1)
#     {
#         "model": UNet(dimensions=2, in_channels=1, out_channels=1,
#                       channels=[8, 16, 32], strides=[2, 2]),
#         "image": torch.rand(1, 1, 64, 64),
#         "label": 0,
#     },
# ]


# class TestComputeOcclusionSensitivity(unittest.TestCase):
#     @parameterized.expand([TEST_CASE_1])
#     def test_shape(self, input_data):
#         result = compute_occlusion_sensitivity(**input_data)
#         print(result.shape)
#         self.assertTupleEqual(result.shape, (64, 64))


if __name__ == "__main__":

    model = densenet121(spatial_dims=2, in_channels=1, out_channels=3)
    image = torch.rand(1, 1, 64, 64)
    label = 0
    compute_occlusion_sensitivity(model, image, label)
    # unittest.main()
