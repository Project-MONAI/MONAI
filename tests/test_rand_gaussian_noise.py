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
from parameterized import parameterized

from monai.transforms import RandGaussianNoise
from tests.utils import NumpyImageTestCase2D, TorchImageTestCase2D


class TestRandGaussianNoise(NumpyImageTestCase2D):
    @parameterized.expand([("test_zero_mean", 0, 0.1), ("test_non_zero_mean", 1, 0.5)])
    def test_correct_results(self, _, mean, std):
        seed = 0
        gaussian_fn = RandGaussianNoise(prob=1.0, mean=mean, std=std)
        gaussian_fn.set_random_state(seed)
        noised = gaussian_fn(self.imt)
        np.random.seed(seed)
        np.random.random()
        expected = self.imt + np.random.normal(mean, np.random.uniform(0, std), size=self.imt.shape)
        np.testing.assert_allclose(expected, noised, atol=1e-5)


class TestRandGaussianNoiseTorch(TorchImageTestCase2D):
    @parameterized.expand([("test_zero_mean", 0, 0.1), ("test_non_zero_mean", 1, 0.5)])
    def test_correct_results(self, _, mean, std):
        seed = 0
        gaussian_fn = RandGaussianNoise(prob=1.0, mean=mean, std=std)
        gaussian_fn.set_random_state(seed)
        noised = gaussian_fn(self.imt)
        np.random.seed(seed)
        np.random.random()
        expected = self.imt + np.random.normal(mean, np.random.uniform(0, std), size=self.imt.shape)
        np.testing.assert_allclose(expected, noised, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
