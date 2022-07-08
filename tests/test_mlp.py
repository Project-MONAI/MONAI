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

import numpy as np
import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.blocks.mlp import MLPBlock

TEST_CASE_MLP = []
for dropout_rate in np.linspace(0, 1, 4):
    for hidden_size in [128, 256, 512, 768]:
        for mlp_dim in [0, 1028, 2048, 3072]:

            test_case = [
                {"hidden_size": hidden_size, "mlp_dim": mlp_dim, "dropout_rate": dropout_rate},
                (2, 512, hidden_size),
                (2, 512, hidden_size),
            ]
            TEST_CASE_MLP.append(test_case)


class TestMLPBlock(unittest.TestCase):
    @parameterized.expand(TEST_CASE_MLP)
    def test_shape(self, input_param, input_shape, expected_shape):
        net = MLPBlock(**input_param)
        with eval_mode(net):
            result = net(torch.randn(input_shape))
            self.assertEqual(result.shape, expected_shape)

    def test_ill_arg(self):
        with self.assertRaises(ValueError):
            MLPBlock(hidden_size=128, mlp_dim=512, dropout_rate=5.0)


if __name__ == "__main__":
    unittest.main()
