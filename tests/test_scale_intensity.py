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

from monai.transforms import ScaleIntensity
from tests.utils import NumpyImageTestCase2D


class TestScaleIntensity(NumpyImageTestCase2D):
    def test_range_scale(self):
        scaler = ScaleIntensity(minv=1.0, maxv=2.0)
        result = scaler(self.imt)
        mina = np.min(self.imt)
        maxa = np.max(self.imt)
        norm = (self.imt - mina) / (maxa - mina)
        expected = (norm * (2.0 - 1.0)) + 1.0
        np.testing.assert_allclose(result, expected)

    def test_factor_scale(self):
        scaler = ScaleIntensity(minv=None, maxv=None, factor=0.1)
        result = scaler(self.imt)
        expected = (self.imt * (1 + 0.1)).astype(np.float32)
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
