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

from parameterized import parameterized

from monai.transforms import GaussianNoise
from tests.utils import NumpyImageTestCase2D


class GaussianNoiseTest(NumpyImageTestCase2D):

    @parameterized.expand([
        ("test_zero_mean", 0, 0.1),
        ("test_non_zero_mean", 1, 0.5)
    ])
    def test_correct_results(self, _, mean, std):
        seed = 42
        gaussian_fn = GaussianNoise(mean=mean, std=std)
        gaussian_fn.set_random_state(seed)
        noised = gaussian_fn(self.imt)
        np.random.seed(seed)
        expected = self.imt + np.random.normal(mean, np.random.uniform(0, std), size=self.imt.shape)
        assert np.allclose(expected, noised)


if __name__ == '__main__':
    unittest.main()
