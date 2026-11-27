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

from monai.transforms.utils import squarepulse
from tests.test_utils import skip_if_no_cuda

# Duty cycle used for value-equality checks across devices / dtypes.
_DUTY = 0.5


class TestSquarePulse(unittest.TestCase):
    @parameterized.expand([(torch.float32,), (torch.float64,)])
    def test_device_and_dtype_follow_floating_input(self, dtype):
        t = torch.linspace(0, 12, 25, dtype=dtype)
        y = squarepulse(t, duty=_DUTY)
        self.assertEqual(y.device, t.device)
        self.assertEqual(y.dtype, dtype)
        self.assertEqual(tuple(y.shape), tuple(t.shape))
        # Square wave is in {-1, 1} for valid duty.
        self.assertTrue(torch.all((y == 1) | (y == -1)))

    def test_integer_input_promotes_to_default_float(self):
        t = torch.arange(0, 25)
        y = squarepulse(t, duty=_DUTY)
        self.assertEqual(y.device, t.device)
        self.assertEqual(y.dtype, torch.get_default_dtype())
        self.assertTrue(torch.all((y == 1) | (y == -1)))

    def test_values_match_cpu_reference(self):
        t = torch.linspace(0, 12, 25)
        y = squarepulse(t, duty=_DUTY)
        # Reference: same math on a fresh CPU float32 buffer (pre-fix default path).
        tmod = torch.remainder(t, 2 * torch.pi)
        expected = torch.where(tmod < _DUTY * 2 * torch.pi, torch.tensor(1.0), torch.tensor(-1.0))
        self.assertTrue(torch.equal(y, expected))


@skip_if_no_cuda
class TestSquarePulseCuda(unittest.TestCase):
    def test_cuda_input_stays_on_cuda(self):
        """Regression: squarepulse must not silently return CPU for CUDA input."""
        t = torch.linspace(0, 12, 25, device="cuda")
        y = squarepulse(t, duty=_DUTY)
        self.assertEqual(y.device.type, "cuda")
        self.assertEqual(y.dtype, t.dtype)
        # Values must match the CPU reference (device is the only difference).
        y_cpu = squarepulse(t.cpu(), duty=_DUTY)
        self.assertTrue(torch.equal(y.cpu(), y_cpu))

    def test_float64_cuda_preserves_dtype_and_device(self):
        t = torch.linspace(0, 12, 25, dtype=torch.float64, device="cuda")
        y = squarepulse(t, duty=_DUTY)
        self.assertEqual(y.device.type, "cuda")
        self.assertEqual(y.dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
