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

import unittest
from unittest import skipUnless

import numpy as np
import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.selfattention import SABlock
from monai.utils import optional_import

einops, has_einops = optional_import("einops")

TEST_CASE_SABLOCK = []
for dropout_rate in np.linspace(0, 1, 4):
    for hidden_size in [360, 480, 600, 768]:
        for num_heads in [4, 6, 8, 12]:

            test_case = [
                {"hidden_size": hidden_size, "num_heads": num_heads, "dropout_rate": dropout_rate},
                (2, 512, hidden_size),
                (2, 512, hidden_size),
            ]
            TEST_CASE_SABLOCK.append(test_case)


class TestResBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_SABLOCK)
    @skipUnless(has_einops, "Requires einops")
    def test_shape(self, input_param, input_shape, expected_shape):
        net = SABlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            SABlock(hidden_size=128, num_heads=12, dropout_rate=6.0)

        with self.assertRaises(ValueError):
            SABlock(hidden_size=620, num_heads=8, dropout_rate=0.4)


if __name__ == "__main__":
    unittest.main()
