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

from monai.inferers import sliding_window_inference

TEST_CASE_1 = [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "constant"]  # 3D small roi

TEST_CASE_2 = [(1, 3, 16, 15, 7), (20, 22, 23), 10, 0.25, "constant"]  # 3D large roi

TEST_CASE_3 = [(1, 3, 15, 7), (2, 6), 1000, 0.25, "constant"]  # 2D small roi, large batch

TEST_CASE_4 = [(1, 3, 16, 7), (80, 50), 7, 0.25, "constant"]  # 2D large roi

TEST_CASE_5 = [(1, 3, 16, 15, 7), (20, 22, 23), 10, 0.5, "constant"]  # 3D large overlap

TEST_CASE_6 = [(1, 3, 16, 7), (80, 50), 7, 0.5, "gaussian"]  # 2D large overlap, gaussian

TEST_CASE_7 = [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "gaussian"]  # 3D small roi, gaussian


class TestSlidingWindowInference(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5, TEST_CASE_6, TEST_CASE_7])
    def test_sliding_window_default(self, image_shape, roi_shape, sw_batch_size, overlap, mode):
        inputs = torch.ones(*image_shape)
        device = torch.device("cpu:0")

        def compute(data):
            return data + 1

        result = sliding_window_inference(inputs.to(device), roi_shape, sw_batch_size, compute, overlap, mode=mode)
        expected_val = np.ones(image_shape, dtype=np.float32) + 1
        self.assertTrue(np.allclose(result.numpy(), expected_val))


if __name__ == "__main__":
    unittest.main()
