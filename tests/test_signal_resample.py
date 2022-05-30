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

import os
import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms.signal.array import SignalResample

TEST_SIGNAL = os.path.join(os.path.dirname(__file__), "testing_data", "signal.npy")


VALID_CASES = [("interpolation", 500, 250), ("polynomial", 500, 250), ("fourier", 500, 250)]


class TestSignalResample(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_multi_channels(self, method, current_sample_rate, target_sample_rate):
        self.assertIsInstance(SignalResample(method, current_sample_rate, target_sample_rate), SignalResample)
        self.assertGreaterEqual(current_sample_rate, target_sample_rate)
        self.assertIn(method, ["interpolation", "polynomial", "fourier"])
        sig = np.load(TEST_SIGNAL)
        Resampled = SignalResample(method, current_sample_rate, target_sample_rate)
        ResampledSignal = Resampled(sig)
        self.assertEqual(len(ResampledSignal), len(sig))

    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_mono_channels(self, method, current_sample_rate, target_sample_rate):
        self.assertIsInstance(SignalResample(method, current_sample_rate, target_sample_rate), SignalResample)
        self.assertGreaterEqual(current_sample_rate, target_sample_rate)
        self.assertIn(method, ["interpolation", "polynomial", "fourier"])
        sig = np.load(TEST_SIGNAL)[0, :]
        Resampled = SignalResample(method, current_sample_rate, target_sample_rate)
        ResampledSignal = Resampled(sig)
        self.assertEqual(len(ResampledSignal), len(sig) / 2)


if __name__ == "__main__":
    unittest.main()
