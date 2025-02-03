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

from monai.transforms import ShiftIntensity
from tests.test_utils import NumpyImageTestCase2D


class TestShiftIntensity(NumpyImageTestCase2D):
    def test_value(self):
        shifter = ShiftIntensity(offset=1.0)
        result = shifter(self.imt)
        expected = self.imt + 1.0
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
