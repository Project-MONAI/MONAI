# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest

import numpy as np

from monai.transforms import RandScaleIntensityFixedMeand
from tests.test_utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestRandScaleIntensityFixedMeand(NumpyImageTestCase2D):
    def test_value(self):
        key = "img"
        for p in TEST_NDARRAYS:
            scaler = RandScaleIntensityFixedMeand(keys=[key], factors=0.5, prob=1.0)
            scaler.set_random_state(seed=0)
            result = scaler({key: p(self.imt)})
            np.random.seed(0)
            # simulate the randomize function of transform
            np.random.random()
            im = self.imt
            mn = im.mean()
            im = im - mn
            expected = (1 + np.random.uniform(low=-0.5, high=0.5)) * im
            expected = expected + mn
            assert_allclose(result[key], p(expected), type_test="tensor", atol=1e-6)


if __name__ == "__main__":
    unittest.main()
