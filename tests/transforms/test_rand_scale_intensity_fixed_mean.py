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
from parameterized import parameterized

from monai.transforms import RandScaleIntensityFixedMean
from tests.test_utils import TEST_NDARRAYS, NumpyImageTestCase2D, assert_allclose


class TestRandScaleIntensity(NumpyImageTestCase2D):
    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_value(self, p):
        scaler = RandScaleIntensityFixedMean(prob=1.0, factors=0.5)
        scaler.set_random_state(seed=0)
        im = p(self.imt)
        result = scaler(im)
        np.random.seed(0)
        # simulate the randomize() of transform
        np.random.random()
        mn = im.mean()
        im = im - mn
        expected = (1 + np.random.uniform(low=-0.5, high=0.5)) * im
        expected = expected + mn
        assert_allclose(result, expected, type_test="tensor", atol=1e-7)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise(self, p):
        scaler = RandScaleIntensityFixedMean(prob=1.0, factors=0.5, channel_wise=True)
        scaler.set_random_state(seed=0)
        im = p(self.imt)
        result = scaler(im)
        np.random.seed(0)
        # simulate the randomize() of transform
        np.random.random()
        channel_num = self.imt.shape[0]
        factor = np.random.uniform(low=-0.5, high=0.5, size=(channel_num,))
        expected = np.stack(
            [
                np.asarray((self.imt[i] - self.imt[i].mean()) * (1 + factor[i]) + self.imt[i].mean())
                for i in range(channel_num)
            ]
        ).astype(np.float32)
        assert_allclose(result, p(expected), atol=1e-4, rtol=1e-4, type_test=False)

    @parameterized.expand([[p] for p in TEST_NDARRAYS])
    def test_channel_wise_preserve_range(self, p):
        scaler = RandScaleIntensityFixedMean(
            prob=1.0, factors=0.5, channel_wise=True, preserve_range=True, fixed_mean=True
        )
        scaler.set_random_state(seed=0)
        im = p(self.imt)
        result = scaler(im)
        # verify output is within input range per channel
        for c in range(self.imt.shape[0]):
            assert float(result[c].min()) >= float(im[c].min()) - 1e-6
            assert float(result[c].max()) <= float(im[c].max()) + 1e-6


if __name__ == "__main__":
    unittest.main()
