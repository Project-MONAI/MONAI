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

from monai.metrics import compute_variance

_device = "cuda:0" if torch.cuda.is_available() else "cpu"
# keep background, 2D Case
TEST_CASE_1 = [  # y_pred (1, 1, 2, 2), expected out (0.0)
    {
        "y_pred": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], device=_device),
        "include_background": True,
        "spatial_map": False,
    },
    [[0.0]],
]

# keep background, 3D Case
TEST_CASE_2 = [  # y_pred (1, 1, 1, 2, 2), expected out (0.0)
    {
        "y_pred": torch.tensor([[[[[1.0, 1.0], [1.0, 1.0]]]]], device=_device),
        "include_background": True,
        "spatial_map": False,
    },
    [[0.0]],
]

# keep background, 1D Case
TEST_CASE_3 = [  # y_pred (3, 1, 3), expected out (0.0)
    {
        "y_pred": torch.tensor([[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]], device=_device),
        "include_background": True,
        "spatial_map": False,
    },
    [[0.0]],
]

# remove background, 1D Case
TEST_CASE_4 = [  # y_pred (3, 1, 3), expected out (0.0)
    {
        "y_pred": torch.tensor(
            [
                [[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]],
                [[4.0, 5.0, 6.0], [1.0, 1.0, 1.0]],
                [[7.0, 8.0, 9.0], [1.0, 1.0, 1.0]],
            ],
            device=_device,
        ),
        "include_background": False,
        "spatial_map": False,
    },
    [[0.0]],
]


class TestComputeVariance(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_value(self, input_data, expected_value):
        result = compute_variance(**input_data)
        print(result)
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
