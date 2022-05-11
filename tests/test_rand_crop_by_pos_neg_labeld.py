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
from copy import deepcopy

import numpy as np
from parameterized import parameterized

from monai.transforms import RandCropByPosNegLabeld
from monai.utils.enums import PostFix
from tests.utils import TEST_NDARRAYS

TESTS = [
    [
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
            PostFix.meta("image"): {"affine": np.eye(3), "shape": "CHWD"},
        },
        (3, 3, 2, 2),
    ],
    [
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
            PostFix.meta("label"): {"affine": np.eye(3), "shape": "CHWD"},
        },
        (3, 2, 2, 2),
    ],
    [
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
            PostFix.meta("extra"): {"affine": np.eye(3), "shape": "CHWD"},
        },
        (3, 2, 2, 2),
    ],
    [
        {
            "keys": ["image", "extra", "label"],
            "label_key": "label",
            "spatial_size": [4, 4, 2],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image_key": None,
            "image_threshold": 0,
            "allow_smaller": True,
        },
        {
            "image": np.zeros([3, 3, 3, 3]) - 1,
            "extra": np.zeros([3, 3, 3, 3]),
            "label": np.ones([3, 3, 3, 3]),
            PostFix.meta("extra"): {"affine": np.eye(3), "shape": "CHWD"},
        },
        (3, 3, 3, 2),
    ],
    [
        {
            "keys": ["image", "extra", "label"],
            "label_key": "label",
            "spatial_size": [4, 4, 4],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image_key": None,
            "image_threshold": 0,
            "allow_smaller": True,
        },
        {
            "image": np.zeros([3, 3, 3, 3]) - 1,
            "extra": np.zeros([3, 3, 3, 3]),
            "label": np.ones([3, 3, 3, 3]),
            PostFix.meta("extra"): {"affine": np.eye(3), "shape": "CHWD"},
        },
        (3, 3, 3, 3),
    ],
]


class TestRandCropByPosNegLabeld(unittest.TestCase):
    @staticmethod
    def convert_data_type(im_type, d, keys=("img", "image", "label")):
        out = deepcopy(d)
        for k, v in out.items():
            if k in keys and isinstance(v, np.ndarray):
                out[k] = im_type(v)
        return out

    @parameterized.expand(TESTS)
    def test_type_shape(self, input_param, input_data, expected_shape):
        for p in TEST_NDARRAYS:
            input_param_mod = self.convert_data_type(p, input_param)
            input_data_mod = self.convert_data_type(p, input_data)
            cropper = RandCropByPosNegLabeld(**input_param_mod)
            cropper.set_random_state(0)
            result = cropper(input_data_mod)
            self.assertListEqual(cropper.spatial_size, input_param["spatial_size"])

            self.assertIsInstance(result, list)

            _len = len(tuple(input_data.keys()))
            self.assertTupleEqual(tuple(result[0].keys())[:_len], tuple(input_data.keys()))
            for k in ("image", "extra", "label"):
                self.assertTupleEqual(result[0][k].shape, expected_shape)
                for i, item in enumerate(result):
                    self.assertEqual(item[PostFix.meta(k)]["patch_index"], i)


if __name__ == "__main__":
    unittest.main()
