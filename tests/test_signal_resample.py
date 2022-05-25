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
from parameterized import parameterized

from monai.transforms.signal.array import SignalResample
from tests.utils import TEST_NDARRAYS, assert_allclose, NumpySignalTestCase

INVALID_CASES = [("wrong_axis", ["s", 1], TypeError), ("not_numbers", "s", TypeError)]

VALID_CASES = [("interpolation", None), ("one_axis", 1), ("many_axis", [0, 1]), ("negative_axis", [0, -1])]


class TestSignalResample(NumpySignalTestCase):
    @parameterized.expand(INVALID_CASES)
    def test_invalid_inputs(self, method, raises):
        with self.assertRaises(raises):
            Resample = SignalResample(method=method)
            Resample(self.sig)

    @parameterized.expand(VALID_CASES)
    def test_correct_results(self, method):
        for p in TEST_NDARRAYS:
            sig = p(self.sig)
            Resample = SignalResample(method='interpolation')
   
            expected = [np.flip(channel, spatial_axis) for channel in self.imt[0]]
            expected = np.stack(expected)
            result = flip(im)
            assert_allclose(result, p(expected))

if __name__ == "__main__":
    unittest.main()
