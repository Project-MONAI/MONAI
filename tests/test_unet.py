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

import torch
from parameterized import parameterized

from monai.networks.nets.unet import UNet


class TestUNET(unittest.TestCase):

    @parameterized.expand([
        [
            {
                'dimensions': 2,
                'in_channels': 1,
                'num_classes': 3,
                'channels': (16, 32, 64),
                'strides': (2, 2),
                'num_res_units': 1,
            },
            torch.randn(16, 1, 32, 32),
            (16, 32, 32),
        ],
        [
            {
                'dimensions': 3,
                'in_channels': 1,
                'num_classes': 3,
                'channels': (16, 32, 64),
                'strides': (2, 2),
                'num_res_units': 1,
            },
            torch.randn(16, 1, 32, 32, 32),
            (16, 32, 32, 32),
        ],
    ])
    def test_shape(self, input_param, input_data, expected_shape):
        result = UNet(**input_param).forward(input_data)[1]
        self.assertEqual(result.shape, expected_shape)


if __name__ == '__main__':
    unittest.main()
