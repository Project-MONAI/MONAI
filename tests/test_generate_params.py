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

from monai.networks.nets import Unet
from monai.optimizers import generate_params

TEST_CASE_1 = [
    {
        "layer_matches": [lambda x: x.model[-1]],
        "lr_values": [1],
    },
    (1, 100),
]

TEST_CASE_2 = [
    {
        "layer_matches": [lambda x: x.model[-1], lambda x: x.model[-2], lambda x: x.model[-3]],
        "lr_values": [1, 2, 3],
    },
    (1, 2, 3, 100),
]

TEST_CASE_3 = [
    {
        "layer_matches": [lambda x: x.model[2][1].conv[0].conv],
        "lr_values": [1],
    },
    (1, 100),
]


class TestGenerateParams(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_lr_values(self, input_param, expected_values):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = Unet(
            dimensions=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=1,
        ).to(device)

        params = generate_params(network=net, **input_param)
        optimizer = torch.optim.Adam(params, 100)

        for param_group, value in zip(optimizer.param_groups, expected_values):
            torch.testing.assert_allclose(param_group["lr"], value)


if __name__ == "__main__":
    unittest.main()
