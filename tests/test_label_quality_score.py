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

from monai.metrics import label_quality_score

_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# keep background, 1D Case
TEST_CASE_1 = [  # y_pred (3, 1, 3), expected out (0.0)
    {
        "y_pred": torch.tensor([[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]], device=_device),
        "y": torch.tensor([[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]], device=_device),
        "include_background": True,
        "spatial_map": False,
        "scalar_reduction": "sum"
    },
    [0.0, 0.0, 0.0],
]

# keep background, 2D Case
TEST_CASE_2 = [  # y_pred (1, 1, 2, 2), expected out (0.0)
    {
        "y_pred": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], device=_device),
        "y": torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]], device=_device),
        "include_background": True,
        "spatial_map": False,
        "scalar_reduction": "sum"
    },
    [0.0],
]

# keep background, 3D Case
TEST_CASE_3 = [  # y_pred (1, 1, 1, 2, 2), expected out (0.0)
    {
        "y_pred": torch.tensor([[[[[1.0, 1.0], [1.0, 1.0]]]]], device=_device),
        "y": torch.tensor([[[[[1.0, 1.0], [1.0, 1.0]]]]], device=_device),
        "include_background": True,
        "spatial_map": False,
        "scalar_reduction": "sum"
    },
    [0.0],
]

class TestLabelQualityScore(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3])
    def test_value(self, input_data, expected_value):
        result = label_quality_score(**input_data)
        print(result.shape)
        print(result)
        print('####')
        np.testing.assert_allclose(result.cpu().numpy(), expected_value, atol=1e-4)

if __name__ == "__main__":
    unittest.main()