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

import torch
from parameterized import parameterized

from monai.transforms import SobelGradients
from tests.utils import assert_allclose

IMAGE = torch.zeros(1, 16, 16, dtype=torch.float32)
IMAGE[0, 8, :] = 1
OUTPUT_3x3 = torch.zeros(2, 16, 16, dtype=torch.float32)
OUTPUT_3x3[0, 7, :] = 2.0
OUTPUT_3x3[0, 9, :] = -2.0
OUTPUT_3x3[0, 7, 0] = OUTPUT_3x3[0, 7, -1] = 1.5
OUTPUT_3x3[0, 9, 0] = OUTPUT_3x3[0, 9, -1] = -1.5
OUTPUT_3x3[1, 7, 0] = OUTPUT_3x3[1, 9, 0] = 0.5
OUTPUT_3x3[1, 8, 0] = 1.0
OUTPUT_3x3[1, 8, -1] = -1.0
OUTPUT_3x3[1, 7, -1] = OUTPUT_3x3[1, 9, -1] = -0.5

TEST_CASE_0 = [IMAGE, {"kernel_size": 3, "dtype": torch.float32}, OUTPUT_3x3]
TEST_CASE_1 = [IMAGE, {"kernel_size": 3, "dtype": torch.float64}, OUTPUT_3x3]

TEST_CASE_KERNEL_0 = [
    {"kernel_size": 3, "dtype": torch.float64},
    torch.tensor([[-0.5, 0.0, 0.5], [-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=torch.float64),
]
TEST_CASE_KERNEL_1 = [
    {"kernel_size": 5, "dtype": torch.float64},
    torch.tensor(
        [
            [-0.25, -0.2, 0.0, 0.2, 0.25],
            [-0.4, -0.5, 0.0, 0.5, 0.4],
            [-0.5, -1.0, 0.0, 1.0, 0.5],
            [-0.4, -0.5, 0.0, 0.5, 0.4],
            [-0.25, -0.2, 0.0, 0.2, 0.25],
        ],
        dtype=torch.float64,
    ),
]
TEST_CASE_KERNEL_2 = [
    {"kernel_size": 7, "dtype": torch.float64},
    torch.tensor(
        [
            [-3.0 / 18.0, -2.0 / 13.0, -1.0 / 10.0, 0.0, 1.0 / 10.0, 2.0 / 13.0, 3.0 / 18.0],
            [-3.0 / 13.0, -2.0 / 8.0, -1.0 / 5.0, 0.0, 1.0 / 5.0, 2.0 / 8.0, 3.0 / 13.0],
            [-3.0 / 10.0, -2.0 / 5.0, -1.0 / 2.0, 0.0, 1.0 / 2.0, 2.0 / 5.0, 3.0 / 10.0],
            [-3.0 / 9.0, -2.0 / 4.0, -1.0 / 1.0, 0.0, 1.0 / 1.0, 2.0 / 4.0, 3.0 / 9.0],
            [-3.0 / 10.0, -2.0 / 5.0, -1.0 / 2.0, 0.0, 1.0 / 2.0, 2.0 / 5.0, 3.0 / 10.0],
            [-3.0 / 13.0, -2.0 / 8.0, -1.0 / 5.0, 0.0, 1.0 / 5.0, 2.0 / 8.0, 3.0 / 13.0],
            [-3.0 / 18.0, -2.0 / 13.0, -1.0 / 10.0, 0.0, 1.0 / 10.0, 2.0 / 13.0, 3.0 / 18.0],
        ],
        dtype=torch.float64,
    ),
]
TEST_CASE_ERROR_0 = [{"kernel_size": 2, "dtype": torch.float32}]


class SobelGradientTests(unittest.TestCase):
    backend = None

    @parameterized.expand([TEST_CASE_0])
    def test_sobel_gradients(self, image, arguments, expected_grad):
        sobel = SobelGradients(**arguments)
        grad = sobel(image)
        assert_allclose(grad, expected_grad)

    @parameterized.expand([TEST_CASE_KERNEL_0, TEST_CASE_KERNEL_1, TEST_CASE_KERNEL_2])
    def test_sobel_kernels(self, arguments, expected_kernel):
        sobel = SobelGradients(**arguments)
        self.assertTrue(sobel.kernel.dtype == expected_kernel.dtype)
        assert_allclose(sobel.kernel, expected_kernel)

    @parameterized.expand([TEST_CASE_ERROR_0])
    def test_sobel_gradients_error(self, arguments):
        with self.assertRaises(ValueError):
            SobelGradients(**arguments)


if __name__ == "__main__":
    unittest.main()
