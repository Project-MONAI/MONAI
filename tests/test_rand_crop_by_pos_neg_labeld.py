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
        {"image": np.zeros([3, 3, 3, 3]) - 1, "extra": np.zeros([3, 3, 3, 3]), "label": np.ones([3, 3, 3, 3])},
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
        {"image": np.zeros([3, 3, 3, 3]) - 1, "extra": np.zeros([3, 3, 3, 3]), "label": np.ones([3, 3, 3, 3])},
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
        {"image": np.zeros([3, 3, 3, 3]) - 1, "extra": np.zeros([3, 3, 3, 3]), "label": np.ones([3, 3, 3, 3])},
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

            self.assertIsInstance(result, list)
            self.assertEqual(len(result), input_param["num_samples"])
            self.assertListEqual(cropper.spatial_size, input_param["spatial_size"])

            with self.assertRaises(NotImplementedError):
                _ = cropper.inverse(result)

            for i, r in enumerate(result):
                inv = cropper.inverse(deepcopy(r))

                for k in ("image", "extra", "label"):
                    self.assertTupleEqual(r[k].shape, expected_shape)
                    self.assertEqual(r[k].meta["patch_index"], i)
                    self.assertEqual(inv[k].shape, input_data[k].shape)

    def test_correct_center(self):
        cropper = RandCropByPosNegLabeld(keys="label", label_key="label", spatial_size=[3, 3])
        cropper.set_random_state(0)
        test_image = {"label": np.asarray([[[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]])}
        result = cropper(test_image)
        np.testing.assert_allclose(result[0]["label"], np.asarray([[[0, 0, 1], [0, 0, 0], [0, 0, 0]]]))


if __name__ == "__main__":
    unittest.main()
