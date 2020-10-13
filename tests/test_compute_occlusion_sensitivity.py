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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2D
TEST_CASE_1 = [
    {
        "model": densenet121(spatial_dims=2, in_channels=1, out_channels=3).to(device),
        "image": torch.rand(1, 1, 64, 64).to(device),
        "label": 0,
    },
    (64, 64),
]


class TestComputeOcclusionSensitivity(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, input_data, expected_shape):
        result = compute_occlusion_sensitivity(**input_data)
        self.assertTupleEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
