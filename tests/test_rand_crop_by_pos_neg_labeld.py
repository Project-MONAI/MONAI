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

from monai.transforms import RandCropByPosNegLabeld

TEST_CASE_0 = [
    {
        "keys": ["image", "extral", "label"],
        "label_key": "label",
        "spatial_size": [-1, 2, 2],
        "pos": 1,
        "neg": 1,
        "num_samples": 2,
        "image_key": None,
        "image_threshold": 0,
    },
    {
        "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "extral": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "affine": np.eye(3),
        "shape": "CHWD",
    },
    list,
    (3, 3, 2, 2),
]

TEST_CASE_1 = [
    {
        "keys": ["image", "extral", "label"],
        "label_key": "label",
        "spatial_size": [2, 2, 2],
        "pos": 1,
        "neg": 1,
        "num_samples": 2,
        "image_key": None,
        "image_threshold": 0,
    },
    {
        "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "extral": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "affine": np.eye(3),
        "shape": "CHWD",
    },
    list,
    (3, 2, 2, 2),
]

TEST_CASE_2 = [
    {
        "keys": ["image", "extral", "label"],
        "label_key": "label",
        "spatial_size": [2, 2, 2],
        "pos": 1,
        "neg": 1,
        "num_samples": 2,
        "image_key": None,
        "image_threshold": 0,
    },
    {
        "image": np.zeros([3, 3, 3, 3]) - 1,
        "extral": np.zeros([3, 3, 3, 3]),
        "label": np.ones([3, 3, 3, 3]),
        "affine": np.eye(3),
        "shape": "CHWD",
    },
    list,
    (3, 2, 2, 2),
]

TEST_CASE_3 = [
    {
        "keys": ["image", "extral", "label"],
        "label_key": "label",
        "spatial_size": [2, 2, 2],
        "pos": 1,
        "neg": 1,
        "num_samples": 1,
        "image_key": None,
        "image_threshold": 0,
    },
    {
        "image": np.zeros([3, 3, 3, 3]) - 1,
        "extral": np.zeros([3, 3, 3, 3]),
        "label": np.ones([3, 3, 3, 3]),
        "affine": np.eye(3),
        "shape": "CHWD",
    },
    dict,
    (3, 2, 2, 2),
]

TESTS = [TEST_CASE_0, TEST_CASE_1, TEST_CASE_2, TEST_CASE_3]


class TestRandCropByPosNegLabeld(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_type_shape(self, input_param, input_data, expected_type, expected_shape):
        result = RandCropByPosNegLabeld(**input_param)(input_data)
        self.assertIsInstance(result, expected_type)
        out = result[0] if isinstance(result, list) else result
        self.assertTupleEqual(out["image"].shape, expected_shape)
        self.assertTupleEqual(out["extral"].shape, expected_shape)
        self.assertTupleEqual(out["label"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
