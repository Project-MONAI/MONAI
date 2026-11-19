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

from monai.transforms import RandSpeckleNoise
from tests.test_utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(("test_sampled_std", p, 0.5, True))
    TESTS.append(("test_fixed_std", p, 0.2, False))
    TESTS.append(("test_zero_std", p, 0.0, False))


class TestRandSpeckleNoise(NumpyImageTestCase2D):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, in_type, std, sample_std):
        seed = 0
        speckle_fn = RandSpeckleNoise(prob=1.0, std=std, sample_std=sample_std)
        speckle_fn.set_random_state(seed)
        im = in_type(self.imt)
        noised = speckle_fn(im)
        if isinstance(im, torch.Tensor):
            self.assertEqual(im.dtype, noised.dtype)
        np.random.seed(seed)
        np.random.random()
        _std = np.random.uniform(0, std) if sample_std else std
        if _std <= 0:
            expected = self.imt
        else:
            expected = self.imt * np.random.gamma(1.0 / _std**2, _std**2, size=self.imt.shape)
        if isinstance(noised, torch.Tensor):
            noised = noised.cpu()
        np.testing.assert_allclose(expected, noised, atol=1e-5)

    def test_prob_zero(self):
        speckle_fn = RandSpeckleNoise(prob=0.0, std=0.5)
        speckle_fn.set_random_state(0)
        out = speckle_fn(self.imt)
        np.testing.assert_allclose(self.imt, np.asarray(out))

    def test_negative_std_raises(self):
        with self.assertRaises(ValueError):
            RandSpeckleNoise(prob=1.0, std=-0.1)


if __name__ == "__main__":
    unittest.main()
