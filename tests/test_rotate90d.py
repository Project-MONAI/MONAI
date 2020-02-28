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

from monai.transforms.composables import Rotate90d
from tests.utils import NumpyImageTestCase2D


class Rotate90Test(NumpyImageTestCase2D):

    def test_rotate90_default(self):
        key = 'test'
        rotate = Rotate90d(keys=key)
        rotated = rotate({key: self.imt})
        expected = np.rot90(self.imt, 1, (1, 2))
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_k(self):
        key = None
        rotate = Rotate90d(keys=key, k=2)
        rotated = rotate({key: self.imt})
        expected = np.rot90(self.imt, 2, (1, 2))
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_axes(self):
        key = ['test']
        rotate = Rotate90d(keys=key, axes=(1, 2))
        rotated = rotate({key[0]: self.imt})
        expected = np.rot90(self.imt, 1, (1, 2))
        self.assertTrue(np.allclose(rotated[key[0]], expected))

    def test_k_axes(self):
        key = ('test',)
        rotate = Rotate90d(keys=key, k=2, axes=(2, 3))
        rotated = rotate({key[0]: self.imt})
        expected = np.rot90(self.imt, 2, (2, 3))
        self.assertTrue(np.allclose(rotated[key[0]], expected))

    def test_no_key(self):
        key = 'unknown'
        rotate = Rotate90d(keys=key)
        with self.assertRaisesRegex(KeyError, ''):
            rotate({'test': self.imt})


if __name__ == '__main__':
    unittest.main()
