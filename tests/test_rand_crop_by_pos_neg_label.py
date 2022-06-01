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
from tests.croppers import CropTest
from parameterized import parameterized

from monai.transforms import RandCropByPosNegLabel
from monai.data.meta_tensor import MetaTensor
from tests.utils import TEST_NDARRAYS, assert_allclose

TESTS = [
    [
        {
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "spatial_size": [2, 2, -1],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "image_threshold": 0,
        },
        {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
        (3, 2, 2, 3),
    ],
    [
        {
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "spatial_size": [2, 2, 2],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "image_threshold": 0,
        },
        {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
        (3, 2, 2, 2),
    ],
    [
        {
            "label": None,
            "spatial_size": [2, 2, 2],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "image_threshold": 0,
        },
        {
            "img": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
        },
        (3, 2, 2, 2),
    ],
    [
        {
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "spatial_size": [4, 4, 2],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "allow_smaller": True,
        },
        {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
        (3, 3, 3, 2),
    ],
    [
        {
            "label": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "spatial_size": [4, 4, 4],
            "pos": 1,
            "neg": 1,
            "num_samples": 2,
            "image": np.random.randint(0, 2, size=[3, 3, 3, 3]),
            "allow_smaller": True,
        },
        {"img": np.random.randint(0, 2, size=[3, 3, 3, 3])},
        (3, 3, 3, 3),
    ],
]


class TestRandCropByPosNegLabel(CropTest):
    @staticmethod
    def convert_data_type(im_type, d, keys=("img", "image", "label")):
        out = deepcopy(d)
        for k, v in out.items():
            if k in keys and isinstance(v, np.ndarray):
                out[k] = im_type(v)
        return out

    @parameterized.expand(TESTS)
    def test_type_shape(self, input_param, input_data, expected_shape):
        base_comparison = None
        for p in TEST_NDARRAYS:
            for q in TEST_NDARRAYS:
                input_param_mod = self.convert_data_type(q, input_param)
                input_data_mod = self.convert_data_type(p, input_data)
                cropper = RandCropByPosNegLabel(**input_param_mod)
                cropper.set_random_state(0)
                result = cropper(**input_data_mod)
                self.assertListEqual(cropper.spatial_size, input_param["spatial_size"])
                for r in result:
                    self.assertIsInstance(r, MetaTensor)
                    self.assertEqual(len(r.applied_operations), 1)

                self.assertIsInstance(result, list)
                self.assertTupleEqual(result[0].shape, expected_shape)

                if base_comparison is None:
                    base_comparison = result

                for b, r in zip(base_comparison, result):
                    assert_allclose(b, r)

                # check inverse
                for r in result:
                    inv = cropper.inverse(r)
                    self.assertIsInstance(inv, MetaTensor)
                    self.assertEqual(inv.applied_operations, [])
                    self.assertEqual(inv.shape, input_data["img"].shape)

if __name__ == "__main__":
    unittest.main()
