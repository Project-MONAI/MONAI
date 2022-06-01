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
from tests.croppers import CropTest
from tests.croppers import CropTest
import numpy as np
from parameterized.parameterized import parameterized

from monai.transforms.croppad.array import RandWeightedCrop
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D, NumpyImageTestCase3D, assert_allclose

from tests.croppers import CropTest
from monai.data.meta_tensor import MetaTensor

def get_data(ndim):
    im_gen = NumpyImageTestCase2D() if ndim == 2 else NumpyImageTestCase3D()
    im_gen.setUp()
    return im_gen.imt[0], im_gen.seg1[0], im_gen.segn[0]


IMT_2D, SEG1_2D, SEGN_2D = get_data(ndim=2)
IMT_3D, SEG1_3D, SEGN_3D = get_data(ndim=3)


TESTS = []
for p in TEST_NDARRAYS:
    for q in TEST_NDARRAYS:
        im = SEG1_2D
        weight = np.zeros_like(im)
        weight[0, 30, 17] = 1.1
        weight[0, 40, 31] = 1
        weight[0, 80, 21] = 1
        TESTS.append(
            [
                "small roi 2d",
                dict(spatial_size=(10, 12), num_samples=3),
                p(im),
                q(weight),
                (1, 10, 12),
                [[80, 21], [30, 17], [40, 31]],
            ]
        )
        im = IMT_2D
        TESTS.append(
            [
                "default roi 2d",
                dict(spatial_size=(10, -1), num_samples=3),
                p(im),
                q(weight),
                (1, 10, 64),
                [[14, 32], [105, 32], [20, 32]],
            ]
        )
        im = SEGN_2D
        weight = np.zeros_like(im)
        weight[0, 30, 17] = 1.1
        weight[0, 10, 1] = 1
        TESTS.append(
            [
                "large roi 2d",
                dict(spatial_size=(10000, 400), num_samples=3),
                p(im),
                q(weight),
                (1, 128, 64),
                [[64, 32], [64, 32], [64, 32]],
            ]
        )
        im = IMT_2D
        weight = np.zeros_like(im)
        weight[0, 30, 17] = np.inf
        weight[0, 10, 1] = -np.inf
        weight[0, 10, 20] = -np.nan
        TESTS.append(
            [
                "bad w 2d",
                dict(spatial_size=(20, 40), num_samples=3),
                p(im),
                q(weight),
                (1, 20, 40),
                [[63, 37], [31, 43], [66, 20]],
            ]
        )
        im = SEG1_3D
        weight = np.zeros_like(im)
        weight[0, 5, 30, 17] = 1.1
        weight[0, 8, 40, 31] = 1
        weight[0, 11, 23, 21] = 1
        TESTS.append(
            [
                "small roi 3d",
                dict(spatial_size=(8, 10, 12), num_samples=3),
                p(im),
                q(weight),
                (1, 8, 10, 12),
                [[11, 23, 21], [5, 30, 17], [8, 40, 31]],
            ]
        )
        im = IMT_3D
        weight = np.zeros_like(im)
        weight[0, 7, 17] = 1.1
        weight[0, 13, 31] = 1.1
        weight[0, 24, 21] = 1
        TESTS.append(
            [
                "default roi 3d",
                dict(spatial_size=(10, -1, -1), num_samples=3),
                p(im),
                q(weight),
                (1, 10, 64, 80),
                [[14, 32, 40], [41, 32, 40], [20, 32, 40]],
            ]
        )
        im = SEGN_3D
        weight = np.zeros_like(im)
        weight[0, 30, 17, 20] = 1.1
        weight[0, 10, 1, 17] = 1
        TESTS.append(
            [
                "large roi 3d",
                dict(spatial_size=(10000, 400, 80), num_samples=3),
                p(im),
                q(weight),
                (1, 48, 64, 80),
                [[24, 32, 40], [24, 32, 40], [24, 32, 40]],
            ]
        )
        im = IMT_3D
        weight = np.zeros_like(im)
        weight[0, 30, 17] = np.inf
        weight[0, 10, 1] = -np.inf
        weight[0, 10, 20] = -np.nan
        TESTS.append(
            [
                "bad w 3d",
                dict(spatial_size=(48, 64, 80), num_samples=3),
                p(im),
                q(weight),
                (1, 48, 64, 80),
                [[24, 32, 40], [24, 32, 40], [24, 32, 40]],
            ]
        )

TEST_MULTI_INVERSE_LIST = [
    [
        (1, 2, 2),
        {"spatial_size": (1, 1), "num_samples": 4, "weight_map": np.ones((1, 2, 2))},
    ],
    [
        (3, 10, 11, 12),
        {"spatial_size": (3, 4, 5), "num_samples": 100, "weight_map": np.ones((3, 10, 11, 12))},
    ],
]

class TestRandWeightedCrop(CropTest):
    Cropper = RandWeightedCrop
    @parameterized.expand(TESTS)
    def test_rand_weighted_crop(self, _, input_params, img, weight, expected_shape, expected_vals):
        crop = RandWeightedCrop(**input_params)
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == input_params["num_samples"])
        assert_allclose(result[0].shape, expected_shape)
        for c, e in zip(crop.centers, expected_vals):
            assert_allclose(c, e, type_test=False)
        # if desired ROI is larger than image, check image is unchanged
        if all(s >= i for i, s in zip(img.shape[1:], input_params["spatial_size"])):
            for res in result:
                self.assertIsInstance(res, MetaTensor)
                assert_allclose(res, img, type_test=False)
                self.assertEqual(len(res.applied_operations), 1)
                # individual inverse
                inv = crop.inverse(deepcopy(res))
                self.assertIsInstance(inv, MetaTensor)
                self.assertEqual(inv.applied_operations, [])
                self.assertTupleEqual(inv.shape, img.shape)


    @parameterized.expand(TEST_MULTI_INVERSE_LIST)
    def test_multi_inverse(self, input_shape, init_params):
        self.multi_inverse(input_shape, init_params)


if __name__ == "__main__":
    unittest.main()
