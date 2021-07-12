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

from monai.transforms import ClassesToIndicesd

TEST_CASE_1 = [
    # test Argmax data
    {"keys": "label", "num_classes": 3, "image_threshold": 0.0},
    {"label": np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])},
    [np.array([0, 4, 8]), np.array([1, 5, 6]), np.array([2, 3, 7])],
]

TEST_CASE_2 = [
    {"keys": "label", "image_key": "image", "num_classes": 3, "image_threshold": 60},
    {
        "label": np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]]),
        "image": np.array([[[132, 1434, 51], [61, 0, 133], [523, 44, 232]]]),
    },
    [np.array([0, 8]), np.array([1, 5, 6]), np.array([3])],
]

TEST_CASE_3 = [
    # test One-Hot data
    {"keys": "label", "image_threshold": 0.0},
    {
        "label": np.array(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            ]
        )
    },
    [np.array([0, 4, 8]), np.array([1, 5, 6]), np.array([2, 3, 7])],
]

TEST_CASE_4 = [
    {"keys": "label", "image_key": "image", "num_classes": None, "image_threshold": 60},
    {
        "label": np.array(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
                [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            ]
        ),
        "image": np.array([[[132, 1434, 51], [61, 0, 133], [523, 44, 232]]]),
    },
    [np.array([0, 8]), np.array([1, 5, 6]), np.array([3])],
]

TEST_CASE_5 = [
    # test output_shape
    {"keys": "label", "indices_postfix": "cls", "num_classes": 3, "image_threshold": 0.0, "output_shape": [3, 3]},
    {"label": np.array([[[0, 1, 2], [2, 0, 1], [1, 2, 0]]])},
    [np.array([[0, 0], [1, 1], [2, 2]]), np.array([[0, 1], [1, 2], [2, 0]]), np.array([[0, 2], [1, 0], [2, 1]])],
]


class TestClassesToIndicesd(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_value(self, input_args, input_data, expected_indices):
        result = ClassesToIndicesd(**input_args)(input_data)
        key_postfix = input_args.get("indices_postfix")
        key_postfix = "_cls_indices" if key_postfix is None else key_postfix
        for i, e in zip(result["label" + key_postfix], expected_indices):
            np.testing.assert_allclose(i, e)


if __name__ == "__main__":
    unittest.main()
