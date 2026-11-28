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

from monai.transforms import RandSpeckleNoised
from tests.test_utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(["test_sampled_std", p, ["img1", "img2"], 0.5, True])
    TESTS.append(["test_fixed_std", p, ["img1", "img2"], 0.2, False])

seed = 0


class TestRandSpeckleNoisedNumpy(NumpyImageTestCase2D):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, in_type, keys, std, sample_std):
        speckle_fn = RandSpeckleNoised(keys=keys, prob=1.0, std=std, sample_std=sample_std, dtype=np.float64)
        speckle_fn.set_random_state(seed)
        noised = speckle_fn({k: in_type(self.imt) for k in keys})
        np.random.seed(seed)
        # simulate the `randomize` function of transform, called once so all keys share one factor
        np.random.random()
        _std = np.random.uniform(0, std) if sample_std else std
        if _std <= 0:
            expected = self.imt
        else:
            expected = self.imt * np.random.gamma(1.0 / _std**2, _std**2, size=self.imt.shape)
        for k in keys:
            if isinstance(noised[k], torch.Tensor):
                noised[k] = noised[k].cpu()
            np.testing.assert_allclose(expected, noised[k], atol=1e-5, rtol=1e-5)
        # identical inputs must get the identical shared speckle factor
        np.testing.assert_array_equal(np.asarray(noised[keys[0]]), np.asarray(noised[keys[1]]))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")
    def test_mps_float32_input(self):
        # MPS does not support float64, so the transform must compute in float32 on MPS inputs.
        keys = ["img1", "img2"]
        speckle_fn = RandSpeckleNoised(keys=keys, prob=1.0, std=0.5, sample_std=False, dtype=None)
        speckle_fn.set_random_state(seed)
        img = torch.full((1, 4, 4), 100.0, dtype=torch.float32, device="mps")
        noised = speckle_fn({k: img for k in keys})
        np.random.seed(seed)
        np.random.random()
        factor = np.random.gamma(1.0 / 0.5**2, 0.5**2, size=img.shape)
        expected = img.cpu().numpy().astype(np.float32) * factor.astype(np.float32)
        for k in keys:
            self.assertEqual(noised[k].device.type, "mps")
            self.assertEqual(noised[k].dtype, torch.float32)
            np.testing.assert_allclose(noised[k].cpu().numpy(), expected, atol=1e-4)
        # identical inputs must get the identical shared speckle factor
        np.testing.assert_array_equal(np.asarray(noised[keys[0]].cpu()), np.asarray(noised[keys[1]].cpu()))


if __name__ == "__main__":
    unittest.main()
