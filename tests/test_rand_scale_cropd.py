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

from monai.transforms import RandScaleCropd
from tests.utils import TEST_NDARRAYS

TEST_CASE_1 = [
    {"keys": "img", "roi_scale": [1.0, 1.0, -1.0], "random_center": True},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 4])},
    (3, 3, 3, 4),
]

TEST_CASE_2 = [
    {"keys": "img", "roi_scale": [1.0, 1.0, 1.0], "random_center": False},
    {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
    (3, 3, 3, 3),
]

TEST_CASE_3 = [
    {"keys": "img", "roi_scale": [0.6, 0.6], "random_center": False},
    {"img": np.array([[[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 2, 3, 2, 0], [0, 1, 2, 1, 0], [0, 0, 0, 0, 0]]])},
]

TEST_CASE_4 = [
    {
        "keys": "img",
        "roi_scale": [0.75, 0.6, 0.5],
        "max_roi_scale": [1.0, -1.0, 0.6],
        "random_center": True,
        "random_size": True,
    },
    {"img": np.random.randint(0, 2, size=[1, 4, 5, 6])},
    (1, 3, 4, 3),
]

TEST_CASE_5 = [
    {"keys": "img", "roi_scale": 0.6, "max_roi_scale": 0.8, "random_center": True, "random_size": True},
    {"img": np.random.randint(0, 2, size=[1, 4, 5, 6])},
    (1, 3, 4, 4),
]

TEST_CASE_6 = [
    {"keys": "img", "roi_scale": 0.2, "max_roi_scale": 0.8, "random_center": True, "random_size": True},
    {"img": np.random.randint(0, 2, size=[1, 4, 5, 6])},
    (1, 3, 2, 4),
]


class TestRandScaleCropd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, input_data, expected_shape):
        results = []
        for p in TEST_NDARRAYS:
            data = {"img": p(input_data["img"])}
            cropper = RandScaleCropd(**input_param)
            cropper.set_random_state(0)
            result = cropper(data)["img"]
            self.assertTupleEqual(result.shape, expected_shape)
            self.assertEqual(type(data["img"]), type(result))
            if isinstance(result, torch.Tensor):
                self.assertEqual(result.device, data["img"].device)
                result = result.cpu().numpy()
            results.append(result)
            if len(results) > 1:
                np.testing.assert_allclose(results[0], results[-1])

    @parameterized.expand([TEST_CASE_3])
    def test_value(self, input_param, input_data):
        results = []
        for p in TEST_NDARRAYS:
            data = {"img": p(input_data["img"])}
            cropper = RandScaleCropd(**input_param)
            cropper.set_random_state(0)
            result = cropper(data)["img"]
            roi = [(2 - i // 2, 2 + i - i // 2) for i in cropper._size]
            self.assertEqual(type(data["img"]), type(result))
            if isinstance(result, torch.Tensor):
                self.assertEqual(result.device, data["img"].device)
                result = result.cpu().numpy()
            np.testing.assert_allclose(result, input_data["img"][:, roi[0][0] : roi[0][1], roi[1][0] : roi[1][1]])
            results.append(result)
            if len(results) > 1:
                np.testing.assert_allclose(results[0], results[-1])

    @parameterized.expand([TEST_CASE_4, TEST_CASE_5, TEST_CASE_6])
    def test_random_shape(self, input_param, input_data, expected_shape):
        results = []
        for p in TEST_NDARRAYS:
            data = {"img": p(input_data["img"])}
            cropper = RandScaleCropd(**input_param)
            cropper.set_random_state(seed=123)
            result = cropper(data)["img"]
            self.assertTupleEqual(result.shape, expected_shape)
            self.assertEqual(type(data["img"]), type(result))
            if isinstance(result, torch.Tensor):
                self.assertEqual(result.device, data["img"].device)
                result = result.cpu().numpy()
            results.append(result)
            if len(results) > 1:
                np.testing.assert_allclose(results[0], results[-1])


if __name__ == "__main__":
    unittest.main()
