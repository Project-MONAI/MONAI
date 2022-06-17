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

from monai.transforms import SignalContinousWavelet

TEST_SIGNAL = os.path.join(os.path.dirname(__file__), "testing_data", "signal.npy")
VALID_CASES = [("mexh", 150, 500)]
EXPECTED_RESULTS = [(6, 150, 2000), (1, 150, 2000)]


class TestSignalRandDrop(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_multi_channels(self, type, length, frequency):
        self.assertIsInstance(SignalContinousWavelet(type, length, frequency), SignalContinousWavelet)
        sig = np.load(TEST_SIGNAL)
        cwt = SignalContinousWavelet(type, length, frequency)
        cwtsignal = cwt(sig)
        self.assertEqual(cwtsignal.shape, EXPECTED_RESULTS[0])

    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_mono_channels(self, type, length, frequency):
        self.assertIsInstance(SignalContinousWavelet(type, length, frequency), SignalContinousWavelet)
        sig = np.load(TEST_SIGNAL)[0, :]
        cwt = SignalContinousWavelet(type, length, frequency)
        cwtsignal = cwt(sig)
        self.assertEqual(cwtsignal.shape, EXPECTED_RESULTS[1])


if __name__ == "__main__":
    unittest.main()
