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

from monai.data import MetaTensor, set_track_meta
from monai.transforms import Rotate90
from tests.utils import (
    TEST_NDARRAYS_ALL,
    NumpyImageTestCase2D,
    NumpyImageTestCase3D,
    assert_allclose,
    test_local_inversion,
)


class TestRotate90(NumpyImageTestCase2D):
    def test_rotate90_default(self):
        rotate = Rotate90()
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            set_track_meta(True)
            rotated = rotate(im)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")
            set_track_meta(False)
            rotated = rotate(im)
            self.assertNotIsInstance(rotated, MetaTensor)
            set_track_meta(True)

    def test_k(self):
        rotate = Rotate90(k=2)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            rotated = rotate(im)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 2, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_spatial_axes(self):
        rotate = Rotate90(spatial_axes=(0, -1))
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            rotated = rotate(im)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, -1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_prob_k_spatial_axes(self):
        rotate = Rotate90(k=2, spatial_axes=(0, 1))
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])

            rotated = rotate(im)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 2, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")


class TestRotate903d(NumpyImageTestCase3D):
    def test_rotate90_default(self):
        rotate = Rotate90()
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            rotated = rotate(im)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_k(self):
        rotate = Rotate90(k=2)
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            rotated = rotate(im)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 2, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_spatial_axes(self):
        rotate = Rotate90(spatial_axes=(0, -1))
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            rotated = rotate(im)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 1, (0, -1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")

    def test_prob_k_spatial_axes(self):
        rotate = Rotate90(k=2, spatial_axes=(0, 1))
        for p in TEST_NDARRAYS_ALL:
            im = p(self.imt[0])
            rotated = rotate(im)
            test_local_inversion(rotate, rotated, im)
            expected = [np.rot90(channel, 2, (0, 1)) for channel in self.imt[0]]
            expected = np.stack(expected)
            assert_allclose(rotated, p(expected), rtol=1.0e-5, atol=1.0e-8, type_test="tensor")


if __name__ == "__main__":
    unittest.main()
