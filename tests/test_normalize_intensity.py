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
from parameterized import parameterized

from monai.transforms import NormalizeIntensity
from tests.utils import NumpyImageTestCase2D

TEST_CASES = [
    [{"nonzero": True}, np.array([0.0, 3.0, 0.0, 4.0]), np.array([0.0, -1.0, 0.0, 1.0])],
    [
        {"subtrahend": np.array([3.5, 3.5, 3.5, 3.5]), "divisor": np.array([0.5, 0.5, 0.5, 0.5]), "nonzero": True},
        np.array([0.0, 3.0, 0.0, 4.0]),
        np.array([0.0, -1.0, 0.0, 1.0]),
    ],
    [{"nonzero": True}, np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])],
    [{"nonzero": False}, np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0])],
    [{"nonzero": False}, np.array([1, 1, 1, 1]), np.array([0.0, 0.0, 0.0, 0.0])],
    [
        {"nonzero": False, "channel_wise": True, "subtrahend": [1, 2, 3]},
        np.ones((3, 2, 2)),
        np.array([[[0.0, 0.0], [0.0, 0.0]], [[-1.0, -1.0], [-1.0, -1.0]], [[-2.0, -2.0], [-2.0, -2.0]]]),
    ],
    [
        {"nonzero": True, "channel_wise": True, "subtrahend": [1, 2, 3], "divisor": [0, 0, 2]},
        np.ones((3, 2, 2)),
        np.array([[[0.0, 0.0], [0.0, 0.0]], [[-1.0, -1.0], [-1.0, -1.0]], [[-1.0, -1.0], [-1.0, -1.0]]]),
    ],
    [
        {"nonzero": True, "channel_wise": False, "subtrahend": 2, "divisor": 0},
        np.ones((3, 2, 2)),
        np.ones((3, 2, 2)) * -1.0,
    ],
    [
        {"nonzero": True, "channel_wise": False, "subtrahend": np.ones((3, 2, 2)) * 0.5, "divisor": 0},
        np.ones((3, 2, 2)),
        np.ones((3, 2, 2)) * 0.5,
    ],
    [
        {"nonzero": True, "channel_wise": True, "subtrahend": np.ones((3, 2, 2)) * 0.5, "divisor": [0, 1, 0]},
        np.ones((3, 2, 2)),
        np.ones((3, 2, 2)) * 0.5,
    ],
]


class TestNormalizeIntensity(NumpyImageTestCase2D):
    def test_default(self):
        normalizer = NormalizeIntensity()
        normalized = normalizer(self.imt.copy())
        self.assertTrue(normalized.dtype == np.float32)
        expected = (self.imt - np.mean(self.imt)) / np.std(self.imt)
        np.testing.assert_allclose(normalized, expected, rtol=1e-5)

    @parameterized.expand(TEST_CASES)
    def test_nonzero(self, input_param, input_data, expected_data):
        normalizer = NormalizeIntensity(**input_param)
        np.testing.assert_allclose(expected_data, normalizer(input_data))

    def test_channel_wise(self):
        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True)
        input_data = np.array([[0.0, 3.0, 0.0, 4.0], [0.0, 4.0, 0.0, 5.0]])
        expected = np.array([[0.0, -1.0, 0.0, 1.0], [0.0, -1.0, 0.0, 1.0]])
        np.testing.assert_allclose(expected, normalizer(input_data))

    def test_value_errors(self):
        input_data = np.array([[0.0, 3.0, 0.0, 4.0], [0.0, 4.0, 0.0, 5.0]])
        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True, subtrahend=[1])
        with self.assertRaises(ValueError):
            normalizer(input_data)
        normalizer = NormalizeIntensity(nonzero=True, channel_wise=True, subtrahend=[1, 2], divisor=[1])
        with self.assertRaises(ValueError):
            normalizer(input_data)


if __name__ == "__main__":
    unittest.main()
