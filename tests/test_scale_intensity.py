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
import torch

from monai.transforms import ScaleIntensity
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D


class TestScaleIntensity(NumpyImageTestCase2D):
    def test_range_scale(self):
        for p in TEST_NDARRAYS:
            scaler = ScaleIntensity(minv=1.0, maxv=2.0)
            result = scaler(p(self.imt))
            mina = self.imt.min()
            maxa = self.imt.max()
            norm = (self.imt - mina) / (maxa - mina)
            expected = p((norm * (2.0 - 1.0)) + 1.0)
            torch.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)

    def test_factor_scale(self):
        for p in TEST_NDARRAYS:
            scaler = ScaleIntensity(minv=None, maxv=None, factor=0.1)
            result = scaler(p(self.imt))
            expected = p((self.imt * (1 + 0.1)).astype(np.float32))
            torch.testing.assert_allclose(result, expected, rtol=1e-7, atol=0)


if __name__ == "__main__":
    unittest.main()
