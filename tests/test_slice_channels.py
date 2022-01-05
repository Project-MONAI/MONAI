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

from monai.networks import one_hot, slice_channels

TEST_CASE_1 = [  # single channel 2D, batch 2, shape (2, 1, 2, 2)
    {"labels": torch.tensor([[[[0, 1], [1, 2]]], [[[2, 1], [1, 0]]]]), "num_classes": 3},
    0,
    (1,),
    (1, 3, 2, 2),
]

TEST_CASE_2 = [  # single channel 1D, batch 2, shape (2, 1, 4)
    {"labels": torch.tensor([[[1, 2, 2, 0]], [[2, 1, 0, 1]]]), "num_classes": 3},
    1,
    (1, 2),
    (2, 2, 4),
    np.array([[[1, 0, 0, 0], [0, 1, 1, 0]], [[0, 1, 0, 1], [1, 0, 0, 0]]]),
]

TEST_CASE_3 = [  # single channel 2D, batch 2, shape (2, 1, 2, 2)
    {"labels": torch.tensor([[[[0, 1], [1, 2]]], [[[2, 1], [1, 0]]]]), "num_classes": 3},
    0,
    (),  # select no channels
    (0, 3, 2, 2),
]


class TestSliceChannels(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, input_data, dim, vals, expected_shape, expected_result=None):
        result = one_hot(**input_data)
        result = slice_channels(result, dim, *vals)

        self.assertEqual(result.shape, expected_shape)
        if expected_result is not None:
            self.assertTrue(np.allclose(expected_result, result.numpy()))


if __name__ == "__main__":
    unittest.main()
