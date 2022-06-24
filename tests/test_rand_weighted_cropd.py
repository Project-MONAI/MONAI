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
from tests.utils import TEST_NDARRAYS_ALL, NumpyImageTestCase2D, NumpyImageTestCase3D, assert_allclose


class TestRandWeightedCrop(NumpyImageTestCase2D):
    def test_rand_weighted_crop_small_roi(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.seg1[0]
                n_samples = 3
                crop = RandWeightedCropd("img", "w", (10, 12), n_samples)
                weight = np.zeros_like(img)
                weight[0, 30, 17] = 1.1
                weight[0, 40, 31] = 1
                weight[0, 80, 21] = 1
                crop.set_random_state(10)
                d = {"img": p(img), "w": q(weight)}
                result = crop(d)
                self.assertTrue(len(result) == n_samples)
                np.testing.assert_allclose(result[0]["img"].shape, (1, 10, 12))
                for c, e in zip(crop.cropper.centers, [[80, 21], [30, 17], [40, 31]]):
                    assert_allclose(c, e, type_test=False)

    def test_rand_weighted_crop_default_roi(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.imt[0]
                n_samples = 3
                crop = RandWeightedCropd("im", "weight", (10, -1), n_samples, "coords")
                weight = np.zeros_like(img)
                weight[0, 30, 17] = 1.1
                weight[0, 40, 31] = 1
                weight[0, 80, 21] = 1
                crop.set_random_state(10)
                data = {"im": p(img), "weight": q(weight), "others": np.nan}
                result = crop(data)
                self.assertTrue(len(result) == n_samples)
                np.testing.assert_allclose(result[0]["im"].shape, (1, 10, 64))
                for c, e in zip(crop.cropper.centers, [[14, 32], [105, 32], [20, 32]]):
                    assert_allclose(c, e, type_test=False)
                assert_allclose(result[1]["im"].meta["crop_center"], [105, 32], type_test=False)

    def test_rand_weighted_crop_large_roi(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.segn[0]
                n_samples = 3
                crop = RandWeightedCropd(("img", "seg"), "weight", (10000, 400), n_samples, "location")
                weight = np.zeros_like(img)
                weight[0, 30, 17] = 1.1
                weight[0, 10, 1] = 1
                crop.set_random_state(10)
                data = {"img": p(img), "seg": p(self.imt[0]), "weight": q(weight)}
                result = crop(data)
                self.assertTrue(len(result) == n_samples)
                np.testing.assert_allclose(result[0]["img"].shape, (1, 128, 64))
                np.testing.assert_allclose(result[0]["seg"].shape, (1, 128, 64))
                for c, e in zip(crop.cropper.centers, [[64, 32], [64, 32], [64, 32]]):
                    assert_allclose(c, e, type_test=False)
                assert_allclose(result[1]["img"].meta["crop_center"], [64, 32], type_test=False)

    def test_rand_weighted_crop_bad_w(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.imt[0]
                n_samples = 3
                crop = RandWeightedCropd(("img", "seg"), "w", (20, 40), n_samples)
                weight = np.zeros_like(img)
                weight[0, 30, 17] = np.inf
                weight[0, 10, 1] = -np.inf
                weight[0, 10, 20] = -np.nan
                crop.set_random_state(10)
                result = crop({"img": p(img), "seg": p(self.segn[0]), "w": q(weight)})
                self.assertTrue(len(result) == n_samples)
                np.testing.assert_allclose(result[0]["img"].shape, (1, 20, 40))
                np.testing.assert_allclose(result[0]["seg"].shape, (1, 20, 40))
                for c, e in zip(crop.cropper.centers, [[63, 37], [31, 43], [66, 20]]):
                    assert_allclose(c, e, type_test=False)


class TestRandWeightedCrop3D(NumpyImageTestCase3D):
    def test_rand_weighted_crop_small_roi(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.seg1[0]
                n_samples = 3
                crop = RandWeightedCropd("img", "w", (8, 10, 12), n_samples)
                weight = np.zeros_like(img)
                weight[0, 5, 30, 17] = 1.1
                weight[0, 8, 40, 31] = 1
                weight[0, 11, 23, 21] = 1
                crop.set_random_state(10)
                result = crop({"img": p(img), "w": q(weight)})
                self.assertTrue(len(result) == n_samples)
                np.testing.assert_allclose(result[0]["img"].shape, (1, 8, 10, 12))
                for c, e in zip(crop.cropper.centers, [[11, 23, 21], [5, 30, 17], [8, 40, 31]]):
                    assert_allclose(c, e, type_test=False)

    def test_rand_weighted_crop_default_roi(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.imt[0]
                n_samples = 3
                crop = RandWeightedCropd(("img", "seg"), "w", (10, -1, -1), n_samples)
                weight = np.zeros_like(img)
                weight[0, 7, 17] = 1.1
                weight[0, 13, 31] = 1.1
                weight[0, 24, 21] = 1
                crop.set_random_state(10)
                result = crop({"img": p(img), "seg": p(self.segn[0]), "w": q(weight)})
                self.assertTrue(len(result) == n_samples)
                np.testing.assert_allclose(result[0]["img"].shape, (1, 10, 64, 80))
                np.testing.assert_allclose(result[0]["seg"].shape, (1, 10, 64, 80))
                for c, e in zip(crop.cropper.centers, [[14, 32, 40], [41, 32, 40], [20, 32, 40]]):
                    assert_allclose(c, e, type_test=False)

    def test_rand_weighted_crop_large_roi(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.segn[0]
                n_samples = 3
                crop = RandWeightedCropd("img", "w", (10000, 400, 80), n_samples)
                weight = np.zeros_like(img)
                weight[0, 30, 17, 20] = 1.1
                weight[0, 10, 1, 17] = 1
                crop.set_random_state(10)
                result = crop({"img": p(img), "w": q(weight)})
                self.assertTrue(len(result) == n_samples)
                np.testing.assert_allclose(result[0]["img"].shape, (1, 48, 64, 80))
                for c, e in zip(crop.cropper.centers, [[24, 32, 40], [24, 32, 40], [24, 32, 40]]):
                    assert_allclose(c, e, type_test=False)

    def test_rand_weighted_crop_bad_w(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.imt[0]
                n_samples = 3
                crop = RandWeightedCropd(("img", "seg"), "w", (48, 64, 80), n_samples)
                weight = np.zeros_like(img)
                weight[0, 30, 17] = np.inf
                weight[0, 10, 1] = -np.inf
                weight[0, 10, 20] = -np.nan
                crop.set_random_state(10)
                result = crop({"img": p(img), "seg": p(self.segn[0]), "w": q(weight)})
                self.assertTrue(len(result) == n_samples)
                np.testing.assert_allclose(result[0]["img"].shape, (1, 48, 64, 80))
                np.testing.assert_allclose(result[0]["seg"].shape, (1, 48, 64, 80))
                for c, e in zip(crop.cropper.centers, [[24, 32, 40], [24, 32, 40], [24, 32, 40]]):
                    assert_allclose(c, e, type_test=False)

    def test_rand_weighted_crop_patch_index(self):
        for p in TEST_NDARRAYS_ALL:
            for q in TEST_NDARRAYS_ALL:
                img = self.imt[0]
                n_samples = 3
                crop = RandWeightedCropd(("img", "seg"), "w", (10, -1, -1), n_samples)
                weight = np.zeros_like(img)
                weight[0, 7, 17] = 1.1
                weight[0, 13, 31] = 1.1
                weight[0, 24, 21] = 1
                crop.set_random_state(10)
                result = crop(
                    {"img": p(img), "seg": p(self.segn[0]), "w": q(weight)}
                )
                self.assertTrue(len(result) == n_samples)
                for c, e in zip(crop.cropper.centers, [[14, 32, 40], [41, 32, 40], [20, 32, 40]]):
                    assert_allclose(c, e, type_test=False)
                for i in range(n_samples):
                    np.testing.assert_allclose(result[i]["img"].shape, (1, 10, 64, 80))
                    np.testing.assert_allclose(result[i]["seg"].shape, (1, 10, 64, 80))
                    np.testing.assert_allclose(result[i]["img"].meta["patch_index"], i)
                    np.testing.assert_allclose(result[i]["seg"].meta["patch_index"], i)


if __name__ == "__main__":
    unittest.main()
