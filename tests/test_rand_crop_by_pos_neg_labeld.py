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
        "keys": ["image", "extra", "label"],
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
        "extra": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "image_meta_dict": {"affine": np.eye(3), "shape": "CHWD"},
    },
    list,
    (3, 3, 2, 2),
]

TEST_CASE_1 = [
    {
        "keys": ["image", "extra", "label"],
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
        "extra": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        "label_meta_dict": {"affine": np.eye(3), "shape": "CHWD"},
    },
    list,
    (3, 2, 2, 2),
]

TEST_CASE_2 = [
    {
        "keys": ["image", "extra", "label"],
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
        "extra": np.zeros([3, 3, 3, 3]),
        "label": np.ones([3, 3, 3, 3]),
        "extra_meta_dict": {"affine": np.eye(3), "shape": "CHWD"},
    },
    list,
    (3, 2, 2, 2),
]


class TestRandCropByPosNegLabeld(unittest.TestCase):
    @parameterized.expand([TEST_CASE_0, TEST_CASE_1, TEST_CASE_2])
    def test_type_shape(self, input_param, input_data, expected_type, expected_shape):
        result = RandCropByPosNegLabeld(**input_param)(input_data)
        self.assertIsInstance(result, expected_type)
        self.assertTupleEqual(result[0]["image"].shape, expected_shape)
        self.assertTupleEqual(result[0]["extra"].shape, expected_shape)
        self.assertTupleEqual(result[0]["label"].shape, expected_shape)
        _len = len(tuple(input_data.keys()))
        self.assertTupleEqual(tuple(result[0].keys())[:_len], tuple(input_data.keys()))
        for i, item in enumerate(result):
            self.assertEqual(item["image_meta_dict"]["patch_index"], i)
            self.assertEqual(item["label_meta_dict"]["patch_index"], i)
            self.assertEqual(item["extra_meta_dict"]["patch_index"], i)


if __name__ == "__main__":
    unittest.main()
