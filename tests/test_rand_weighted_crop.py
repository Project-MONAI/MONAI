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

from monai.transforms.croppad.array import RandWeightedCrop
from tests.utils import NumpyImageTestCase2D, NumpyImageTestCase3D


class TestRandWeightedCrop2D(NumpyImageTestCase2D):
    def test_rand_weighted_crop_small_roi(self):
        img = self.seg1[0]
        n_samples = 3
        crop = RandWeightedCrop((10, 12), n_samples)
        weight = np.zeros_like(img)
        weight[0, 30, 17] = 1.1
        weight[0, 40, 31] = 1
        weight[0, 80, 21] = 1
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == n_samples)
        np.testing.assert_allclose(result[0].shape, (1, 10, 12))
        np.testing.assert_allclose(np.asarray(crop.centers), [[80, 21], [30, 17], [40, 31]])

    def test_rand_weighted_crop_default_roi(self):
        img = self.imt[0]
        n_samples = 3
        crop = RandWeightedCrop((10, -1), n_samples)
        weight = np.zeros_like(img)
        weight[0, 30, 17] = 1.1
        weight[0, 40, 31] = 1
        weight[0, 80, 21] = 1
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == n_samples)
        np.testing.assert_allclose(result[0].shape, (1, 10, 64))
        np.testing.assert_allclose(np.asarray(crop.centers), [[14, 32], [105, 32], [20, 32]])

    def test_rand_weighted_crop_large_roi(self):
        img = self.segn[0]
        n_samples = 3
        crop = RandWeightedCrop((10000, 400), n_samples)
        weight = np.zeros_like(img)
        weight[0, 30, 17] = 1.1
        weight[0, 10, 1] = 1
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == n_samples)
        np.testing.assert_allclose(result[0].shape, (1, 128, 64))
        np.testing.assert_allclose(np.asarray(crop.centers), [[64, 32], [64, 32], [64, 32]])
        for res in result:
            np.testing.assert_allclose(res, self.segn[0])

    def test_rand_weighted_crop_bad_w(self):
        img = self.imt[0]
        n_samples = 3
        crop = RandWeightedCrop((20, 40), n_samples)
        weight = np.zeros_like(img)
        weight[0, 30, 17] = np.inf
        weight[0, 10, 1] = -np.inf
        weight[0, 10, 20] = -np.nan
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == n_samples)
        np.testing.assert_allclose(result[0].shape, (1, 20, 40))
        np.testing.assert_allclose(np.asarray(crop.centers), [[63, 37], [31, 43], [66, 20]])


class TestRandWeightedCrop(NumpyImageTestCase3D):
    def test_rand_weighted_crop_small_roi(self):
        img = self.seg1[0]
        n_samples = 3
        crop = RandWeightedCrop((8, 10, 12), n_samples)
        weight = np.zeros_like(img)
        weight[0, 5, 30, 17] = 1.1
        weight[0, 8, 40, 31] = 1
        weight[0, 11, 23, 21] = 1
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == n_samples)
        np.testing.assert_allclose(result[0].shape, (1, 8, 10, 12))
        np.testing.assert_allclose(np.asarray(crop.centers), [[11, 23, 21], [5, 30, 17], [8, 40, 31]])

    def test_rand_weighted_crop_default_roi(self):
        img = self.imt[0]
        n_samples = 3
        crop = RandWeightedCrop((10, -1, -1), n_samples)
        weight = np.zeros_like(img)
        weight[0, 7, 17] = 1.1
        weight[0, 13, 31] = 1.1
        weight[0, 24, 21] = 1
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == n_samples)
        np.testing.assert_allclose(result[0].shape, (1, 10, 64, 80))
        np.testing.assert_allclose(np.asarray(crop.centers), [[14, 32, 40], [41, 32, 40], [20, 32, 40]])

    def test_rand_weighted_crop_large_roi(self):
        img = self.segn[0]
        n_samples = 3
        crop = RandWeightedCrop((10000, 400, 80), n_samples)
        weight = np.zeros_like(img)
        weight[0, 30, 17, 20] = 1.1
        weight[0, 10, 1, 17] = 1
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == n_samples)
        np.testing.assert_allclose(result[0].shape, (1, 48, 64, 80))
        np.testing.assert_allclose(np.asarray(crop.centers), [[24, 32, 40], [24, 32, 40], [24, 32, 40]])
        for res in result:
            np.testing.assert_allclose(res, self.segn[0])

    def test_rand_weighted_crop_bad_w(self):
        img = self.imt[0]
        n_samples = 3
        crop = RandWeightedCrop((48, 64, 80), n_samples)
        weight = np.zeros_like(img)
        weight[0, 30, 17] = np.inf
        weight[0, 10, 1] = -np.inf
        weight[0, 10, 20] = -np.nan
        crop.set_random_state(10)
        result = crop(img, weight)
        self.assertTrue(len(result) == n_samples)
        np.testing.assert_allclose(result[0].shape, (1, 48, 64, 80))
        np.testing.assert_allclose(np.asarray(crop.centers), [[24, 32, 40], [24, 32, 40], [24, 32, 40]])


if __name__ == "__main__":
    unittest.main()
