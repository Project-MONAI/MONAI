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

from monai.transforms import IdentityIntensity
from tests.utils import NumpyImageTestCase2D


class TestIdentityIntensity(NumpyImageTestCase2D):
    def test_IdentityIntensity(self):
        img = self.imt
        identity = IdentityIntensity()
        self.assertTrue(np.allclose(img, identity(img)))


if __name__ == "__main__":
    unittest.main()
