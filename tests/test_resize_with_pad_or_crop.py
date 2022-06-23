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

from monai.transforms import ResizeWithPadOrCrop
from tests.utils import TEST_NDARRAYS_ALL

TEST_CASES = [
    [{"spatial_size": [15, 8, 8], "mode": "constant"}, (3, 8, 8, 4), (3, 15, 8, 8)],
    [
        {"spatial_size": [15, 4, 8], "mode": "constant", "method": "end", "constant_values": 1},
        (3, 8, 8, 4),
        (3, 15, 4, 8),
    ],
    [{"spatial_size": [15, 4, -1], "mode": "constant"}, (3, 8, 8, 4), (3, 15, 4, 4)],
    [{"spatial_size": [15, 4, -1], "mode": "reflect"}, (3, 8, 8, 4), (3, 15, 4, 4)],
    [{"spatial_size": [-1, -1, -1], "mode": "reflect"}, (3, 8, 8, 4), (3, 8, 8, 4)],
]


class TestResizeWithPadOrCrop(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_pad_shape(self, input_param, input_shape, expected_shape):
        for p in TEST_NDARRAYS_ALL:
            if isinstance(p(0), torch.Tensor) and (
                "constant_values" in input_param or input_param["mode"] == "reflect"
            ):
                continue
            paddcroper = ResizeWithPadOrCrop(**input_param)
            result = paddcroper(p(np.zeros(input_shape)))
            np.testing.assert_allclose(result.shape, expected_shape)
            result = paddcroper(p(np.zeros(input_shape)), mode="constant")
            np.testing.assert_allclose(result.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
