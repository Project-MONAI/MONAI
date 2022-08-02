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

import unittest

import numpy as np

from monai.transforms import RandScaleIntensityd
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestRandScaleIntensityd(NumpyImageTestCase2D):
    def test_value(self):
        key = "img"
        for p in TEST_NDARRAYS:
            scaler = RandScaleIntensityd(keys=[key], factors=0.5, prob=1.0)
            scaler.set_random_state(seed=0)
            result = scaler({key: p(self.imt)})
            np.random.seed(0)
            # simulate the randomize function of transform
            np.random.random()
            expected = (self.imt * (1 + np.random.uniform(low=-0.5, high=0.5))).astype(np.float32)
            assert_allclose(result[key], p(expected), type_test="tensor")


if __name__ == "__main__":
    unittest.main()
