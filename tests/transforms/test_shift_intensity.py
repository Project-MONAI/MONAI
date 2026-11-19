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
        for offset, expected in ((1, [1, 101, 255]), (-1, [0, 99, 254]), (1.0, [1, 101, 255]), (-1.0, [0, 99, 254])):
            with self.subTest(offset=offset):
                result = ShiftIntensity(offset=offset, safe=True)(image)
                torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.uint8))

    def test_safe_integer_offset_clamps_signed_image(self):
        image = torch.tensor([-128, 0, 127], dtype=torch.int8)
        result = ShiftIntensity(offset=1, safe=True)(image)
        torch.testing.assert_close(result, torch.tensor([-127, 1, 127], dtype=torch.int8))
        result = ShiftIntensity(offset=-1, safe=True)(image)
        torch.testing.assert_close(result, torch.tensor([-128, -1, 126], dtype=torch.int8))

    def test_safe_int64_preserves_large_integer_precision(self):
        image = torch.tensor([2**60 + 1, torch.iinfo(torch.int64).max], dtype=torch.int64)
        result = ShiftIntensity(offset=1, safe=True)(image)
        torch.testing.assert_close(result, torch.tensor([2**60 + 2, torch.iinfo(torch.int64).max], dtype=torch.int64))
        negative = torch.tensor([-(2**60 + 1), torch.iinfo(torch.int64).min], dtype=torch.int64)
        result = ShiftIntensity(offset=-1, safe=True)(negative)
        torch.testing.assert_close(
            result, torch.tensor([-(2**60 + 2), torch.iinfo(torch.int64).min], dtype=torch.int64)
        )

    def test_safe_int64_offset_beyond_scalar_range(self):
        image = torch.tensor([2**62], dtype=torch.int64)
        result = ShiftIntensity(offset=float(-3 * 2**62), safe=True)(image)
        torch.testing.assert_close(result, torch.tensor([torch.iinfo(torch.int64).min], dtype=torch.int64))

    def test_safe_bool_input(self):
        image = torch.tensor([False, True], dtype=torch.bool)
        for offset, expected in ((0, [False, True]), (1, [True, True]), (-1, [False, False])):
            with self.subTest(offset=offset):
                result = ShiftIntensity(offset=offset, safe=True)(image)
                torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.bool))

    def test_safe_uint64_rejected_explicitly(self):
        image = torch.tensor([2**63 + 1], dtype=torch.uint64)
        with self.assertRaisesRegex(TypeError, "does not support torch.uint64"):
            ShiftIntensity(offset=1, safe=True)(image)

    def test_safe_integer_nonfinite_offsets_rejected(self):
        image = torch.tensor([1, 2], dtype=torch.int64)
        for offset in (
            float("inf"),
            float("-inf"),
            float("nan"),
            np.float32("inf"),
            np.float32("-inf"),
            np.float32("nan"),
        ):
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, "finite offsets"):
                ShiftIntensity(offset=offset, safe=True)(image)

    def test_value(self):
        shifter = ShiftIntensity(offset=1.0)
        result = shifter(self.imt)
        expected = self.imt + 1.0
        np.testing.assert_allclose(result, expected)


if __name__ == "__main__":
    unittest.main()
