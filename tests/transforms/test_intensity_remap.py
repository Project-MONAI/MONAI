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

import torch
from parameterized import parameterized

from monai.transforms import IntensityRemap, RandIntensityRemap
from tests.test_utils import assert_allclose

# input ranges with a negative, zero and positive minimum
TEST_RANGES = [(-1000.0, 3000.0), (0.0, 255.0), (10.0, 20.0)]


class TestIntensityRemap(unittest.TestCase):
    @parameterized.expand(TEST_RANGES)
    def test_output_range(self, low, high):
        img = torch.linspace(low, high, 500).reshape(1, 10, 50)
        remap = IntensityRemap(kernel_size=10, slope=0.7)
        remap.set_random_state(seed=0)
        result = remap(img)
        self.assertEqual(result.shape, img.shape)
        assert_allclose(result.min(), low, type_test=False, atol=1e-3, rtol=1e-5)
        assert_allclose(result.max(), high, type_test=False, atol=1e-3, rtol=1e-5)


class TestRandIntensityRemap(unittest.TestCase):
    @parameterized.expand(TEST_RANGES)
    def test_output_range(self, low, high):
        img = torch.stack([torch.linspace(low, high, 500).reshape(10, 50)] * 2)
        remap = RandIntensityRemap(prob=1.0, kernel_size=10, channel_wise=True)
        remap.set_random_state(seed=0)
        result = remap(img)
        self.assertEqual(result.shape, img.shape)
        for c in range(img.shape[0]):
            assert_allclose(result[c].min(), low, type_test=False, atol=1e-3, rtol=1e-5)
            assert_allclose(result[c].max(), high, type_test=False, atol=1e-3, rtol=1e-5)

    @parameterized.expand([[True], [False]])
    def test_set_random_state(self, channel_wise):
        img = torch.stack([torch.linspace(0.0, 100.0, 500).reshape(10, 50)] * 2)
        results = []
        for _ in range(2):
            remap = RandIntensityRemap(prob=1.0, kernel_size=10, channel_wise=channel_wise)
            remap.set_random_state(seed=0)
            results.append(remap(img))
        assert_allclose(results[0], results[1], type_test=False)

    def test_prob_zero(self):
        img = torch.linspace(-1.0, 1.0, 500).reshape(1, 10, 50)
        remap = RandIntensityRemap(prob=0.0)
        assert_allclose(remap(img), img, type_test=False)


if __name__ == "__main__":
    unittest.main()
