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

import numpy as np

from monai.transforms.croppad.dictionary import RandWeightedCropd
from copy import deepcopy
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D, NumpyImageTestCase3D, assert_allclose

from parameterized import parameterized

def get_data(ndim):
    im_gen = NumpyImageTestCase2D() if ndim == 2 else NumpyImageTestCase3D()
    im_gen.setUp()
    return im_gen.imt[0], im_gen.seg1[0], im_gen.segn[0]


IMT_2D, SEG1_2D, SEGN_2D = get_data(ndim=2)
IMT_3D, SEG1_3D, SEGN_3D = get_data(ndim=3)

TESTS = []
for p in TEST_NDARRAYS:
    for q in TEST_NDARRAYS:
        im = IMT_2D
        weight = np.zeros_like(im)
        weight[0, 30, 17] = 1.1
        weight[0, 40, 31] = 1
        weight[0, 80, 21] = 1
        TESTS.append(
            [
                "small roi 2d",
                dict(keys="img", w_key="w", spatial_size=(10, 12), num_samples=3),
                {"img": p(im), "w": q(weight)},
                (1, 10, 12),
                [[80, 21], [30, 17], [40, 31]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17] = 1.1
        weight[0, 40, 31] = 1
        weight[0, 80, 21] = 1
        TESTS.append(
            [
                "default roi 2d",
                dict(keys="img", w_key="w", spatial_size=(10, -1), num_samples=3),
                {"img": p(im), "w": q(weight), "others": np.nan},
                (1, 10, 64),
                [[14, 32], [105, 32], [20, 32]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17] = 1.1
        weight[0, 10, 1] = 1
        TESTS.append(
            [
                "large roi 2d",
                dict(keys=("img", "seg"), w_key="weight", spatial_size=(10000, 400), num_samples=3),
                {"img": p(im), "seg": p(SEGN_2D), "weight": q(weight)},
                (1, 128, 64),
                [[64, 32], [64, 32], [64, 32]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17] = np.inf
        weight[0, 10, 1] = -np.inf
        weight[0, 10, 20] = -np.nan
        TESTS.append(
            [
                "bad w roi 2d",
                dict(keys=("img", "seg"), w_key="w", spatial_size=(20, 40), num_samples=3),
                {"img": p(im), "seg": p(SEGN_2D), "w": q(weight)},
                (1, 20, 40),
                [[63, 37], [31, 43], [66, 20]],
            ]
        )

        im = IMT_3D
        weight = np.zeros_like(im)
        weight[0, 5, 30, 17] = 1.1
        weight[0, 8, 40, 31] = 1
        weight[0, 11, 23, 21] = 1
        TESTS.append(
            [
                "small roi 3d",
                dict(keys="img", w_key="w", spatial_size=(8, 10, 12), num_samples=3),
                {"img": p(im), "w": q(weight)},
                (1, 8, 10, 12),
                [[11, 23, 21], [5, 30, 17], [8, 40, 31]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 5, 30, 17] = 1.1
        weight[0, 8, 40, 31] = 1
        weight[0, 11, 23, 21] = 1
        TESTS.append(
            [
                "default roi 3d",
                dict(keys=("img", "seg"), w_key="w", spatial_size=(10, -1, -1), num_samples=3),
                {"img": p(im), "seg": p(SEGN_3D), "w": q(weight)},
                (1, 10,64,80),
                [[14, 32, 40], [41, 32, 40], [20, 32, 40]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17, 20] = 1.1
        weight[0, 10, 1, 17] = 1
        TESTS.append(
            [
                "large roi 3d",
                dict(keys="img", w_key="w", spatial_size=(10000, 400, 80), num_samples=3),
                {"img": p(im), "w": q(weight)},
                (1, 48, 64, 80),
                [[24, 32, 40], [24, 32, 40], [24, 32, 40]],
            ]
        )

        weight = np.zeros_like(im)
        weight[0, 30, 17] = np.inf
        weight[0, 10, 1] = -np.inf
        weight[0, 10, 20] = -np.nan
        TESTS.append(
            [
                "bad w roi 3d",
                dict(keys=("img", "seg"), w_key="w", spatial_size=(48,64,80), num_samples=3),
                {"img": p(im), "seg": p(SEGN_3D), "w": q(weight)},
                (1, 48, 64, 80),
                [[24, 32, 40], [24, 32, 40], [24, 32, 40]],
            ]
        )

class TestRandWeightedCrop(unittest.TestCase):
    @parameterized.expand(TESTS)
    def test_rand_weighted_cropd(self, _, init_params, input_data, expected_shape, expected_centers):
        crop = RandWeightedCropd(**init_params)
        crop.set_random_state(10)
        result = crop(input_data)
        self.assertTrue(len(result) == init_params["num_samples"])

        # inverse not implemented for list of output
        with self.assertRaises(NotImplementedError):
            _ = crop.inverse(result)
        for i, (r, e) in enumerate(zip(result, expected_centers)):

            inv = crop.inverse(deepcopy(r))

            for k in crop.keys:
                np.testing.assert_allclose(r[k].shape, expected_shape)
                assert_allclose(r[k].meta["crop_center"], e, type_test=False)
                self.assertEqual(r[k].meta["patch_index"], i)
                # check inverse shape
                self.assertTupleEqual(inv[k].shape, input_data[k].shape)



if __name__ == "__main__":
    unittest.main()
