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

from monai.transforms import RandLocSpatialCropd
from tests.utils import NumpyImageTestCase2D


class TestRandLocSpatialCropd(NumpyImageTestCase2D):

    def test_2d(self):
        roi_size = (10, 10)
        key = 'test'
        patch_transform = RandLocSpatialCropd(keys='test', roi_size=roi_size)
        patch = patch_transform({key: self.imt[0]})
        self.assertTrue(np.allclose(patch[key].shape[1:], roi_size))

    def test_sync(self):
        roi_size = (4, 4)
        key_1, key_2 = 'foo', 'bar'
        rand_image = np.random.rand(3, 10, 10)
        patch_transform = RandLocSpatialCropd(keys=(key_1, key_2), roi_size=roi_size)
        patch = patch_transform({key_1: rand_image, key_2: rand_image})
        self.assertTrue(np.allclose(patch[key_1], patch[key_2]))


if __name__ == '__main__':
    unittest.main()
