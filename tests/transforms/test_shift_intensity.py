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

from monai.transforms import ShiftIntensity
from tests.test_utils import NumpyImageTestCase2D


class TestShiftIntensity(NumpyImageTestCase2D):
    def test_safe_integer_offset_clamps_unsigned_image(self):
        image = torch.tensor([0, 100, 255], dtype=torch.uint8)
        for offset, expected in (
            (1, [1, 101, 255]),
            (-1, [0, 99, 254]),
            (1.0, [1, 101, 255]),
            (-1.0, [0, 99, 254]),
        ):
            with self.subTest(offset=offset):
                result = ShiftIntensity(offset=offset, safe=True)(image)
                torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.uint8))

    def test_safe_integer_offset_clamps_signed_image(self):
        image = torch.tensor([-128, 0, 127], dtype=torch.int8)
        result = ShiftIntensity(offset=1, safe=True)(image)
        torch.testing.assert_close(result, torch.tensor([-127, 1, 127], dtype=torch.int8))
        result = ShiftIntensity(offset=-1, safe=True)(image)
        torch.testing.assert_close(result, torch.tensor([-128, -1, 126], dtype=torch.int8))

    def test_value(self):
        shifter = ShiftIntensity(offset=1.0)
        result = shifter(self.imt)
        expected = self.imt + 1.0
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
