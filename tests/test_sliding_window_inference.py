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

import numpy as np
import torch
from parameterized import parameterized

from monai.utils.sliding_window_inference import sliding_window_inference

TEST_CASE_1 = [(1, 3, 16, 15, 7), (4, 10, 7), 3]  # 3D small roi

TEST_CASE_2 = [(1, 3, 16, 15, 7), (20, 22, 23), 10]  # 3D large roi

TEST_CASE_3 = [(1, 3, 15, 7), (2, 6), 1000]  # 2D small roi, large batch

TEST_CASE_4 = [(1, 3, 16, 7), (80, 50), 7]  # 2D large roi


class TestSlidingWindowInference(unittest.TestCase):

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_sliding_window_default(self, image_shape, roi_shape, sw_batch_size):
        inputs = np.ones(image_shape)
        device = torch.device("cpu:0")

        def compute(data):
            return data.to(device) + 1

        result = sliding_window_inference(inputs, roi_shape, sw_batch_size, compute, device)
        expected_val = np.ones(image_shape, dtype=np.float32) + 1
        self.assertTrue(np.allclose(result.numpy(), expected_val))


if __name__ == '__main__':
    unittest.main()
