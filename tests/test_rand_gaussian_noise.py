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
import torch
from parameterized import parameterized

from monai.transforms import RandGaussianNoise
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(("test_zero_mean", p, 0, 0.1))
    TESTS.append(("test_non_zero_mean", p, 1, 0.5))


class TestRandGaussianNoise(NumpyImageTestCase2D):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, im_type, mean, std):
        seed = 0
        gaussian_fn = RandGaussianNoise(prob=1.0, mean=mean, std=std)
        gaussian_fn.set_random_state(seed)
        im = im_type(self.imt)
        noised = gaussian_fn(im)
        np.random.seed(seed)
        np.random.random()
        expected = self.imt + np.random.normal(mean, np.random.uniform(0, std), size=self.imt.shape)
        self.assertEqual(type(im), type(noised))
        if isinstance(noised, torch.Tensor):
            noised = noised.cpu()
        np.testing.assert_allclose(expected, noised, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
