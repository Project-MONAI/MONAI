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

from monai.transforms import CenterSpatialCrop
from tests.utils import TEST_NDARRAYS

TEST_SHAPES, TEST_VALUES = [], []
for p in TEST_NDARRAYS:
    TEST_SHAPES.append([{"roi_size": [2, 2, -1]}, p(np.random.randint(0, 2, size=[3, 3, 3, 3])), (3, 2, 2, 3)])

    TEST_SHAPES.append([{"roi_size": [2, 2, 2]}, p(np.random.randint(0, 2, size=[3, 3, 3, 3])), (3, 2, 2, 2)])

    TEST_VALUES.append([
        {"roi_size": [2, 2]},
        p(np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])),
        p(np.array([[[1, 2], [2, 3]]])),
    ])

class TestCenterSpatialCrop(unittest.TestCase):
    @parameterized.expand(TEST_SHAPES)
    def test_shape(self, input_param, input_data, expected_shape):
        result = CenterSpatialCrop(**input_param)(input_data)
        np.testing.assert_allclose(result.shape, expected_shape)

    @parameterized.expand(TEST_VALUES)
    def test_value(self, input_param, input_data, expected_value):
        result = CenterSpatialCrop(**input_param)(input_data)
        torch.testing.assert_allclose(result, expected_value, rtol=1e-7, atol=0)


if __name__ == "__main__":
    unittest.main()
