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

from monai.transforms.transforms import UniformRandomPatch
from tests.utils import NumpyImageTestCase2D


class TestUniformRandomPatch(NumpyImageTestCase2D):

    def test_2d(self):
        patch_spatial_size = (10, 10)
        patch_transform = UniformRandomPatch(patch_spatial_size=patch_spatial_size)
        patch = patch_transform(self.imt[0])
        self.assertTrue(np.allclose(patch.shape[1:], patch_spatial_size))


if __name__ == '__main__':
    unittest.main()
