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

from monai.transforms import RandUltrasoundDepthGain
from tests.test_utils import TEST_NDARRAYS, NumpyImageTestCase2D

TESTS = []
for p in TEST_NDARRAYS:
    TESTS.append(("test_gain_range", p, (-1.0, 1.0), 1))
    TESTS.append(("test_fixed_slope", p, (0.5, 0.5), 1))
    TESTS.append(("test_last_axis", p, (-1.0, 1.0), -1))


class TestRandUltrasoundDepthGain(NumpyImageTestCase2D):
    @parameterized.expand(TESTS)
    def test_correct_results(self, _, in_type, gain_range, depth_axis):
        seed = 0
        gain_fn = RandUltrasoundDepthGain(prob=1.0, gain_range=gain_range, depth_axis=depth_axis)
        gain_fn.set_random_state(seed)
        im = in_type(self.imt)
        out = gain_fn(im)
        if isinstance(im, torch.Tensor):
            self.assertEqual(im.dtype, out.dtype)
        np.random.seed(seed)
        np.random.random()
        slope = np.random.uniform(gain_range[0], gain_range[1])
        axis = depth_axis if depth_axis >= 0 else self.imt.ndim + depth_axis
        gain_shape = [1] * self.imt.ndim
        gain_shape[axis] = self.imt.shape[axis]
        profile = np.exp(slope * (np.linspace(0.0, 1.0, self.imt.shape[axis]) - 0.5)).reshape(gain_shape)
        expected = self.imt * profile
        if isinstance(out, torch.Tensor):
            out = out.cpu()
        np.testing.assert_allclose(expected, out, atol=1e-5)

    def test_gain_increases_with_depth(self):
        gain_fn = RandUltrasoundDepthGain(prob=1.0, gain_range=(1.0, 1.0), depth_axis=1)
        gain_fn.set_random_state(0)
        out = gain_fn(np.ones((1, 1, 8, 4), dtype=np.float32))
        row_means = np.asarray(out).mean(axis=(0, 2, 3))
        self.assertTrue(np.all(np.diff(row_means) > 0))

    def test_prob_zero(self):
        gain_fn = RandUltrasoundDepthGain(prob=0.0, gain_range=(-1.0, 1.0))
        gain_fn.set_random_state(0)
        out = gain_fn(self.imt)
        np.testing.assert_allclose(self.imt, np.asarray(out))

    def test_invalid_depth_axis_raises(self):
        gain_fn = RandUltrasoundDepthGain(prob=1.0, gain_range=(0.5, 0.5), depth_axis=5)
        gain_fn.set_random_state(0)
        with self.assertRaises(ValueError):
            gain_fn(self.imt)


if __name__ == "__main__":
    unittest.main()
