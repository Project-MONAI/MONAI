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

from monai.inferers import SlidingWindowInferer, sliding_window_inference

TEST_CASE_0 = [(1, 3, 16, 15, 7), (4, -1, 7), 3, 0.25, "constant", torch.device("cpu:0")]  # 3D small roi

TEST_CASE_1 = [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "constant", torch.device("cpu:0")]  # 3D small roi

TEST_CASE_2 = [(1, 3, 16, 15, 7), (20, 22, 23), 10, 0.25, "constant", torch.device("cpu:0")]  # 3D large roi

TEST_CASE_3 = [(1, 3, 15, 7), (2, 6), 1000, 0.25, "constant", torch.device("cpu:0")]  # 2D small roi, large batch

TEST_CASE_4 = [(1, 3, 16, 7), (80, 50), 7, 0.25, "constant", torch.device("cpu:0")]  # 2D large roi

TEST_CASE_5 = [(1, 3, 16, 15, 7), (20, 22, 23), 10, 0.5, "constant", torch.device("cpu:0")]  # 3D large overlap

TEST_CASE_6 = [(1, 3, 16, 7), (80, 50), 7, 0.5, "gaussian", torch.device("cpu:0")]  # 2D large overlap, gaussian

TEST_CASE_7 = [(1, 3, 16, 15, 7), (4, 10, 7), 3, 0.25, "gaussian", torch.device("cpu:0")]  # 3D small roi, gaussian

TEST_CASE_8 = [
    (1, 3, 16, 15, 7),
    (4, 10, 7),
    3,
    0.25,
    "gaussian",
    torch.device("cuda:0"),
]  # test inference on gpu if availabe

TEST_CASE_9 = [(1, 3, 16, 15, 7), (4, 1, 7), 3, 0.25, "constant", torch.device("cpu:0")]  # 3D small roi


class TestSlidingWindowInference(unittest.TestCase):
    @parameterized.expand(
        [
            TEST_CASE_0,
            TEST_CASE_1,
            TEST_CASE_2,
            TEST_CASE_3,
            TEST_CASE_4,
            TEST_CASE_5,
            TEST_CASE_6,
            TEST_CASE_7,
            TEST_CASE_8,
            TEST_CASE_9,
        ]
    )
    def test_sliding_window_default(self, image_shape, roi_shape, sw_batch_size, overlap, mode, device):
        inputs = torch.ones(*image_shape)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu:0")

        def compute(data):
            return data + 1

        result = sliding_window_inference(inputs.to(device), roi_shape, sw_batch_size, compute, overlap, mode=mode)
        np.testing.assert_string_equal(device.type, result.device.type)
        expected_val = np.ones(image_shape, dtype=np.float32) + 1
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap, mode)(inputs.to(device), compute)
        np.testing.assert_string_equal(device.type, result.device.type)
        expected_val = np.ones(image_shape, dtype=np.float32) + 1
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    def test_default_device(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 3, 16, 15, 7)).to(device=device)
        roi_shape = (4, 10, 7)
        sw_batch_size = 10

        def compute(data):
            return data + 1

        result = sliding_window_inference(inputs, roi_shape, sw_batch_size, compute)
        np.testing.assert_string_equal(inputs.device.type, result.device.type)
        expected_val = np.ones((1, 3, 16, 15, 7), dtype=np.float32) + 1
        np.testing.assert_allclose(result.cpu().numpy(), expected_val)

    def test_sigma(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 7, 7)).to(device=device)
        roi_shape = (3, 3)
        sw_batch_size = 10

        class _Pred:
            add = 1

            def compute(self, data):
                self.add += 1
                return data + self.add

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            _Pred().compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="constant",
            sigma_scale=1.0,
        )

        expected = np.array(
            [
                [
                    [
                        [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                        [3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000, 3.0000],
                        [3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333, 3.3333],
                        [3.6667, 3.6667, 3.6667, 3.6667, 3.6667, 3.6667, 3.6667],
                        [4.3333, 4.3333, 4.3333, 4.3333, 4.3333, 4.3333, 4.3333],
                        [4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000],
                        [5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000, 5.0000],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)
        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            _Pred().compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="gaussian",
            sigma_scale=1.0,
        )
        expected = np.array(
            [
                [
                    [
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                        [3.3271625, 3.3271623, 3.3271623, 3.3271623, 3.3271623, 3.3271623, 3.3271625],
                        [3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377, 3.6728377],
                        [4.3271623, 4.3271623, 4.3271627, 4.3271627, 4.3271627, 4.3271623, 4.3271623],
                        [4.513757, 4.513757, 4.513757, 4.513757, 4.513757, 4.513757, 4.513757],
                        [4.9999995, 5.0, 5.0, 5.0, 5.0, 5.0, 4.9999995],
                    ]
                ]
            ]
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=1.0)(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="gaussian", sigma_scale=[1.0, 1.0])(
            inputs, _Pred().compute
        )
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

    def test_cval(self):
        device = "cuda" if torch.cuda.is_available() else "cpu:0"
        inputs = torch.ones((1, 1, 3, 3)).to(device=device)
        roi_shape = (5, 5)
        sw_batch_size = 10

        def compute(data):
            return data + data.sum()

        result = sliding_window_inference(
            inputs,
            roi_shape,
            sw_batch_size,
            compute,
            overlap=0.5,
            padding_mode="constant",
            cval=-1,
            mode="constant",
            sigma_scale=1.0,
        )
        expected = np.ones((1, 1, 3, 3)) * -6.0
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)

        result = SlidingWindowInferer(roi_shape, sw_batch_size, overlap=0.5, mode="constant", cval=-1)(inputs, compute)
        np.testing.assert_allclose(result.cpu().numpy(), expected, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
