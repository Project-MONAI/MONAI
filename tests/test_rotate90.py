# Copyright 2020 MONAI Consortium
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

from monai.transforms.transforms import Rotate90
from tests.utils import NumpyImageTestCase2D


class Rotate90Test(NumpyImageTestCase2D):

    def test_rotate90_default(self):
        rotate = Rotate90()
        rotated = rotate(self.imt)
        expected = np.rot90(self.imt, 1, (1, 2))
        self.assertTrue(np.allclose(rotated, expected))

    def test_k(self):
        rotate = Rotate90(k=2)
        rotated = rotate(self.imt)
        expected = np.rot90(self.imt, 2, (1, 2))
        self.assertTrue(np.allclose(rotated, expected))

    def test_axes(self):
        rotate = Rotate90(axes=(1, 2))
        rotated = rotate(self.imt)
        expected = np.rot90(self.imt, 1, (1, 2))
        self.assertTrue(np.allclose(rotated, expected))

    def test_k_axes(self):
        rotate = Rotate90(k=2, axes=(2, 3))
        rotated = rotate(self.imt)
        expected = np.rot90(self.imt, 2, (2, 3))
        self.assertTrue(np.allclose(rotated, expected))


if __name__ == '__main__':
    unittest.main()
