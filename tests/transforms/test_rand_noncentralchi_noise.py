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
import torch
from parameterized import parameterized

from monai.transforms import RandNonCentralChiNoise
from tests.test_utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(("test_zero_mean", p, 0, 0.1))
    TESTS.append(("test_non_zero_mean", p, 1, 0.5))


class TestRandNonCentralChiNoise(NumpyImageTestCase2D):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, in_type, mean, std):
        seed = 0
        degrees_of_freedom = 64  #64 is common due to 32 channel head coil
        noise_fn = RandNonCentralChiNoise(prob=1.0, mean=mean, std=std, degrees_of_freedom=degrees_of_freedom)
        noise_fn.set_random_state(seed)
        im = in_type(self.imt)
        noised = noise_fn(im)
        if isinstance(im, torch.Tensor):
            self.assertEqual(im.dtype, noised.dtype)
        np.random.seed(seed)
        np.random.random()
        _std = np.random.uniform(0, std)

        noise_shape = (degrees_of_freedom, *self.imt.shape)
        all_noises = np.random.normal(mean, _std, size=noise_shape).astype(np.float32)
        all_noises[0] += self.imt
        sum_sq = np.sum(all_noises**2, axis=0)
        expected = np.sqrt(sum_sq)

        if isinstance(noised, torch.Tensor):
            noised = noised.cpu()
        np.testing.assert_allclose(expected, noised, atol=1e-5)

    @parameterized.expand(TESTS)
    def test_correct_results_dof2(self, _, in_type, mean, std):
        """
        Test with k=2 (the Rician case)
        """
        seed = 0
        degrees_of_freedom = 2
        noise_fn = RandNonCentralChiNoise(prob=1.0, mean=mean, std=std, degrees_of_freedom=degrees_of_freedom)
        noise_fn.set_random_state(seed)
        im = in_type(self.imt)
        noised = noise_fn(im)
        if isinstance(im, torch.Tensor):
            self.assertEqual(im.dtype, noised.dtype)

        np.random.seed(seed)
        np.random.random()  # for prob
        _std = np.random.uniform(0, std)  # for sample_std
        noise_shape = (degrees_of_freedom, *self.imt.shape)
        all_noises = np.random.normal(mean, _std, size=noise_shape).astype(np.float32)
        all_noises[0] += self.imt
        sum_sq = np.sum(all_noises**2, axis=0)
        expected = np.sqrt(sum_sq)

        if isinstance(noised, torch.Tensor):
            noised = noised.cpu()
        np.testing.assert_allclose(expected, noised, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
