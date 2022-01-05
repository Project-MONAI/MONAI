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

from monai.networks import one_hot
from monai.utils import sample_slices
from tests.utils import TEST_NDARRAYS, assert_allclose

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


class TestSampleSlices(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_shape(self, input_data, dim, vals, expected_shape, expected_result=None):
        onehot = one_hot(**input_data)
        for p in TEST_NDARRAYS:
            result = sample_slices(p(onehot), dim, *vals)

            self.assertEqual(result.shape, expected_shape)
            if expected_result is not None:
                assert_allclose(p(expected_result), result)


if __name__ == "__main__":
    unittest.main()
