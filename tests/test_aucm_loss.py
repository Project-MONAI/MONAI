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

from monai.losses import AUCMLoss

TEST_CASES = [
    [   # shape: (2, 2, 2, 2), (2, 2, 2, 2)
        {
            "y_pred": torch.tensor([[[[0, 1.0], [0, 1.0]], [[0, 1.0], [0, 1.0]]], [[[0, 1.0], [0, 1.0]], [[0, 1.0], [0, 1.0]]]]).cuda(),
            "y_true": torch.tensor([[[[0, 1.0], [0, 1.0]], [[0, 1.0], [0, 1.0]]], [[[0, 1.0], [0, 1.0]], [[0, 1.0], [0, 1.0]]]]).cuda(),
        },
        -3.0,
    ],
]


class TestAUCMLoss(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_result(self, input_data, expected_val):
        loss = AUCMLoss(num_classes=2)
        result = loss(**input_data)
        np.testing.assert_allclose(result.detach().cpu().numpy(), expected_val, atol=1e-4, rtol=1e-4)
    
    def test_single_result(self):
        loss = AUCMLoss(num_classes=1)
        y_p = torch.tensor([[[[1, 0], [0, 1.0]]], [[[0, 0], [0, 1.0]]]])
        y_t = torch.tensor([[[[0, 0], [0, 1.0]]], [[[0, 0], [0, 1.0]]]])
        result = loss(y_p, y_t)
        np.testing.assert_allclose(result.detach().cpu().numpy(), 0.125, atol=1e-4, rtol=1e-4)

    def test_with_cuda(self):
        loss = AUCMLoss(num_classes=1, gpu=True)
        i = torch.tensor([[[[1, 0], [0, 1.0]]], [[[0, 0], [0, 1.0]]]])
        j = torch.tensor([[[[0, 0], [0, 1.0]]], [[[0, 0], [0, 1.0]]]])
        if torch.cuda.is_available():
            i = i.cuda()
            j = j.cuda()
        output = loss(i, j)
        np.testing.assert_allclose(output.detach().cpu().numpy(), 0.125, atol=1e-4, rtol=1e-4)

    def test_ill_shape(self):
        loss = AUCMLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss(torch.ones((2, 2, 2)), torch.ones((2, 2, 2, 2)))
            
    def test_ill_shape_pred_equal_numclasses(self):
        loss = AUCMLoss(num_classes=1)
        with self.assertRaisesRegex(ValueError, ""):
            loss(torch.ones((2, 2, 2)), torch.ones((2, 1, 2)))

    def test_ill_shape_pred_equal_numclasses(self):
        loss = AUCMLoss(num_classes=1)
        with self.assertRaisesRegex(ValueError, ""):
            loss(torch.ones((2, 1, 2)), torch.ones((2, 2, 2)))

if __name__ == "__main__":
    unittest.main()