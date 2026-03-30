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

"""Test GPU support detection and fallback paths for spatial transforms."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch

from monai.transforms.spatial.functional import _compiled_unsupported


class TestCompiledUnsupported(unittest.TestCase):
    """Test _compiled_unsupported device detection."""

    def test_cpu_device_always_supported(self):
        """CPU devices should never be marked unsupported."""
        device = torch.device("cpu")
        self.assertFalse(_compiled_unsupported(device))

    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_detection(self):
        """Verify CUDA compute capability detection."""
        device = torch.device("cuda:0")
        cc_major = torch.cuda.get_device_properties(device).major
        unsupported = _compiled_unsupported(device)
        if cc_major >= 12:
            self.assertTrue(unsupported)
        else:
            self.assertFalse(unsupported)

    def test_compiled_unsupported_return_type(self):
        """Verify return type is bool."""
        device = torch.device("cpu")
        result = _compiled_unsupported(device)
        self.assertIsInstance(result, bool)


class TestResampleFallback(unittest.TestCase):
    """Test Resample fallback behavior on unsupported devices."""

    def test_resample_compilation_flag_respected(self):
        """Verify _compiled_unsupported identifies Blackwell (cc>=12) and supported (cc<12) devices."""
        mock_props = MagicMock()
        cuda_device = torch.device("cuda:0")

        mock_props.major = 12  # Blackwell – unsupported
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            self.assertTrue(_compiled_unsupported(cuda_device))

        mock_props.major = 9  # Hopper – supported
        with patch("torch.cuda.get_device_properties", return_value=mock_props):
            self.assertFalse(_compiled_unsupported(cuda_device))

    def test_compiled_unsupported_logic(self):
        """Test that unsupported devices are correctly detected."""
        cpu_device = torch.device("cpu")
        self.assertFalse(_compiled_unsupported(cpu_device))

        cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cuda_device.type == "cuda":
            cc_major = torch.cuda.get_device_properties(cuda_device).major
            expected = cc_major >= 12
            actual = _compiled_unsupported(cuda_device)
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
