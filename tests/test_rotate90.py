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

from monai.transforms import Rotate90
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D


class TestRotate90(NumpyImageTestCase2D):
    def test_rotate90_default(self):
        rotate = Rotate90()
        for p in TEST_NDARRAYS:
            rotated = rotate(p(self.imt[0]))
            expected = []
            for channel in self.imt[0]:
                expected.append(np.rot90(channel, 1, (0, 1)))
            expected = np.stack(expected)
            self.assertTrue(np.allclose(rotated, expected))

    def test_k(self):
        rotate = Rotate90(k=2)
        for p in TEST_NDARRAYS:
            rotated = rotate(p(self.imt[0]))
            expected = []
            for channel in self.imt[0]:
                expected.append(np.rot90(channel, 2, (0, 1)))
            expected = np.stack(expected)
            self.assertTrue(np.allclose(rotated, expected))

    def test_spatial_axes(self):
        rotate = Rotate90(spatial_axes=(0, -1))
        for p in TEST_NDARRAYS:
            rotated = rotate(p(self.imt[0]))
            expected = []
            for channel in self.imt[0]:
                expected.append(np.rot90(channel, 1, (0, -1)))
            expected = np.stack(expected)
            self.assertTrue(np.allclose(rotated, expected))

    def test_prob_k_spatial_axes(self):
        rotate = Rotate90(k=2, spatial_axes=(0, 1))
        for p in TEST_NDARRAYS:
            rotated = rotate(p(self.imt[0]))
            expected = []
            for channel in self.imt[0]:
                expected.append(np.rot90(channel, 2, (0, 1)))
            expected = np.stack(expected)
            self.assertTrue(np.allclose(rotated, expected))


if __name__ == "__main__":
    unittest.main()
