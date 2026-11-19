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

from monai.transforms import RandUltrasoundDepthGaind
from tests.test_utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(["test_gain_range", p, ["img1", "img2"], (-1.0, 1.0), 1])
    TESTS.append(["test_fixed_slope", p, ["img1", "img2"], (0.5, 0.5), 1])

seed = 0


class TestRandUltrasoundDepthGaindNumpy(NumpyImageTestCase2D):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, in_type, keys, gain_range, depth_axis):
        gain_fn = RandUltrasoundDepthGaind(
            keys=keys, prob=1.0, gain_range=gain_range, depth_axis=depth_axis, dtype=np.float64
        )
        gain_fn.set_random_state(seed)
        noised = gain_fn({k: in_type(self.imt) for k in keys})
        np.random.seed(seed)
        for k in keys:
            # simulate the `randomize` function of transform
            np.random.random()
            slope = np.random.uniform(gain_range[0], gain_range[1])
            axis = depth_axis if depth_axis >= 0 else self.imt.ndim + depth_axis
            gain_shape = [1] * self.imt.ndim
            gain_shape[axis] = self.imt.shape[axis]
            profile = np.exp(slope * (np.linspace(0.0, 1.0, self.imt.shape[axis]) - 0.5)).reshape(gain_shape)
            expected = self.imt * profile
            if isinstance(noised[k], torch.Tensor):
                noised[k] = noised[k].cpu()
            np.testing.assert_allclose(expected, noised[k], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
