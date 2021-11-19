# Copyright 2020 - 2021 MONAI Consortium
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

from monai.networks.nets import DiNTS, DintsSearchSpace

TEST_CASES_3D = [
    [
        {
            "channel_mul": 1,
            "num_blocks": 6,
            "num_depths": 3,
            "device": 'cpu',
            "use_downsample": False,
        },
        {
            "in_channels": 1,
            "num_classes": 3,
            "act_name": 'RELU',
            "norm_name": 'INSTANCE',
            "use_downsample": False,
        },
        (7, 1, 32, 32, 16),
        (7, 3, 32, 32, 16),
    ],
    [
        {
            "channel_mul": 0.5,
            "num_blocks": 7,
            "num_depths": 4,
            "device": 'cuda',
            "use_downsample": True,
        },
        {
            "in_channels": 2,
            "num_classes": 2,
            "act_name": 'PRELU',
            "norm_name": 'BATCH',
            "use_downsample": True,
        },
        (7, 2, 32, 32, 16),
        (7, 2, 32, 32, 16),
    ],
]


class TestDints(unittest.TestCase):
    @parameterized.expand(TEST_CASES_3D)
    def test_factorized_increase_3d(self, dints_grid_params, dints_params, input_shape, expected_shape):
        grid = DintsSearchSpace(**dints_grid_params)
        dints_params['dints_space'] = grid
        net = DiNTS(**dints_params).to(dints_grid_params['device'])
        code2out = net.dints_space.arch_code2out
        cell_ops = net.dints_space.cell_ops
        num_blocks = net.num_blocks
        num_depths = net.num_depths
        node_a = torch.ones((num_blocks + 1, num_depths)).to(dints_grid_params['device'])
        code_a = torch.ones((num_blocks, len(code2out))).to(dints_grid_params['device'])
        code_c = torch.ones((num_blocks, len(code2out), cell_ops)).to(dints_grid_params['device'])
        result = net(torch.randn(input_shape).to(dints_grid_params['device']), arch_code=[node_a, code_a, code_c])[-1]
        self.assertEqual(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
