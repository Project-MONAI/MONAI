# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANy KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms import SignalRemoveFrequency

TEST_SIGNAL = os.path.join(os.path.dirname(__file__), "testing_data", "signal.npy")
VALID_CASES = [(60, 1, 500)]


class TestSignalRandDrop(unittest.TestCase):
    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_multi_channels(self, frequency, quality_factor, sampling_freq):
        self.assertIsInstance(SignalRemoveFrequency(frequency, quality_factor, sampling_freq), SignalRemoveFrequency)
        sig = np.load(TEST_SIGNAL)
        t = sig.shape[1] / sampling_freq
        composite_sig = sig + np.sin(2 * np.pi * frequency * t)
        freqremove = SignalRemoveFrequency(frequency, quality_factor, sampling_freq)
        freqremovesignal = freqremove(composite_sig)
        y = np.fft.fft(composite_sig) / composite_sig.shape[1]
        y = y[: composite_sig.shape[1] // 2]
        y2 = np.fft.fft(freqremovesignal) / freqremovesignal.shape[1]
        y2 = y2[: freqremovesignal.shape[1] // 2]
        self.assertEqual(composite_sig.shape[1], sig.shape[1])
        self.assertAlmostEqual(y.all(), y2.all())

    @parameterized.expand(VALID_CASES)
    def test_correct_parameters_mono_channels(self, frequency, quality_factor, sampling_freq):
        self.assertIsInstance(SignalRemoveFrequency(frequency, quality_factor, sampling_freq), SignalRemoveFrequency)
        sig = np.load(TEST_SIGNAL)[0, :]
        t = len(sig) / sampling_freq
        composite_sig = sig + np.sin(2 * np.pi * frequency * t)
        freqremove = SignalRemoveFrequency(frequency, quality_factor, sampling_freq)
        freqremovesignal = freqremove(composite_sig)
        y = np.fft.fft(composite_sig) / len(composite_sig)
        y = y[: len(composite_sig) // 2]
        y2 = np.fft.fft(freqremovesignal) / len(freqremovesignal)
        y2 = y2[: len(freqremovesignal) // 2]
        self.assertEqual(len(composite_sig), len(sig))
        self.assertAlmostEqual(y.all(), y2.all())


if __name__ == "__main__":
    unittest.main()
