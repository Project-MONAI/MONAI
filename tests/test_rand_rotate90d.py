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

from monai.transforms import RandRotate90d
from tests.utils import NumpyImageTestCase2D


class TestRandRotate90d(NumpyImageTestCase2D):
    def test_default(self):
        key = None
        rotate = RandRotate90d(keys=key)
        rotate.set_random_state(123)
        rotated = rotate({key: self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.rot90(channel, 0, (0, 1)))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_k(self):
        key = "test"
        rotate = RandRotate90d(keys=key, max_k=2)
        rotate.set_random_state(234)
        rotated = rotate({key: self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.rot90(channel, 0, (0, 1)))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_spatial_axes(self):
        key = "test"
        rotate = RandRotate90d(keys=key, spatial_axes=(0, 1))
        rotate.set_random_state(234)
        rotated = rotate({key: self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.rot90(channel, 0, (0, 1)))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_prob_k_spatial_axes(self):
        key = "test"
        rotate = RandRotate90d(keys=key, prob=1.0, max_k=2, spatial_axes=(0, 1))
        rotate.set_random_state(234)
        rotated = rotate({key: self.imt[0]})
        expected = []
        for channel in self.imt[0]:
            expected.append(np.rot90(channel, 1, (0, 1)))
        expected = np.stack(expected)
        self.assertTrue(np.allclose(rotated[key], expected))

    def test_no_key(self):
        key = "unknown"
        rotate = RandRotate90d(keys=key, prob=1.0, max_k=2, spatial_axes=(0, 1))
        with self.assertRaisesRegex(KeyError, ""):
            rotate({"test": self.imt[0]})


if __name__ == "__main__":
    unittest.main()
