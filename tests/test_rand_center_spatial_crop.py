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

from monai.transforms import RandCenterSpatialCrop
from tests.utils import NumpyImageTestCase2D


class TestRandCenterSpatialCrop(NumpyImageTestCase2D):

    def test_2d(self):
        roi_size = (10, 10)
        patch_transform = RandCenterSpatialCrop(roi_size=roi_size)
        patch = patch_transform(self.imt[0])
        self.assertTrue(np.allclose(patch.shape[1:], roi_size))


if __name__ == '__main__':
    unittest.main()
