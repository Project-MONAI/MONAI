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

from monai.transforms import RandGaussianNoised
from tests.utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(["test_zero_mean", p, ["img1", "img2"], 0, 0.1])
    TESTS.append(["test_non_zero_mean", p, ["img1", "img2"], 1, 0.5])

seed = 0


class TestRandGaussianNoised(NumpyImageTestCase2D):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, im_type, keys, mean, std):
        gaussian_fn = RandGaussianNoised(keys=keys, prob=1.0, mean=mean, std=std, dtype=np.float64)
        gaussian_fn.set_random_state(seed)
        im = im_type(self.imt)
        noised = gaussian_fn({k: im for k in keys})
        np.random.seed(seed)
        # simulate the randomize() of transform
        np.random.random()
        noise = np.random.normal(mean, np.random.uniform(0, std), size=self.imt.shape)
        for k in keys:
            expected = self.imt + noise
            self.assertEqual(type(im), type(noised[k]))
            if isinstance(noised[k], torch.Tensor):
                noised[k] = noised[k].cpu()
            np.testing.assert_allclose(expected, noised[k], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
