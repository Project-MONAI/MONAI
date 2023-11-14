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

from __future__ import annotations

import unittest

import torch
from parameterized import parameterized

from monai.networks import eval_mode
from monai.networks.nets.controlnet import ControlNet

TEST_CASES = [
    [
        {
            "spatial_dims": 2,
            "in_channels": 1,
            "num_res_blocks": 1,
            "num_channels": (8, 8, 8),
            "attention_levels": (False, False, True),
            "num_head_channels": 8,
            "norm_num_groups": 8,
            "conditioning_embedding_in_channels": 1,
            "conditioning_embedding_num_channels": (8, 8),
        },
        6,
        (1, 8, 4, 4),
    ]
]


class TestControlNet(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_shape_unconditioned_models(self, input_param, expected_num_down_blocks_residuals, expected_shape):
        net = ControlNet(**input_param)
        with eval_mode(net):
            result = net.forward(
                torch.rand((1, 1, 16, 16)), torch.randint(0, 1000, (1,)).long(), torch.rand((1, 1, 32, 32))
            )
            self.assertEqual(len(result[0]), expected_num_down_blocks_residuals)
            self.assertEqual(result[1].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
