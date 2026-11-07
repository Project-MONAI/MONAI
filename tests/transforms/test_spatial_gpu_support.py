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


def _get_max_cc() -> int:
    """Return the max compute capability (major*100+minor) the _C extension was compiled for,
    or 0 if the extension is not available (no build info)."""
    try:
        from monai._C import max_compute_capability

        return int(max_compute_capability())
    except (ImportError, AttributeError):
        return 0


def _has_sm120_support() -> bool:
    """Return True if the compiled _C extension includes sm_120 (cc 12.0) support."""
    return _get_max_cc() >= 1200


class TestCompiledUnsupported(unittest.TestCase):
    """Test _compiled_unsupported device detection."""

    def test_cpu_device_always_supported(self):
        """CPU devices should never be marked unsupported."""
        device = torch.device("cpu")
        self.assertFalse(_compiled_unsupported(device))

    @unittest.skipIf(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device_detection(self):
        """Verify CUDA compute capability detection against compiled arch list."""
        device = torch.device("cuda:0")
        cc = torch.cuda.get_device_properties(device)
        device_cc = cc.major * 100 + cc.minor
        max_cc = _get_max_cc()
        if max_cc == 0:
            # No build info — rely on heuristic
            expected = cc.major >= 12
        else:
            expected = device_cc > max_cc
        self.assertEqual(_compiled_unsupported(device), expected)

    def test_compiled_unsupported_return_type(self):
        """Verify return type is bool."""
        device = torch.device("cpu")
        result = _compiled_unsupported(device)
        self.assertIsInstance(result, bool)


class TestResampleFallback(unittest.TestCase):
    """Test Resample fallback behavior on unsupported devices."""

    def test_resample_compilation_flag_respected(self):
        """Verify _compiled_unsupported compares device CC against compiled arch list."""
        mock_props = MagicMock()
        cuda_device = torch.device("cuda:0")

        max_cc = _get_max_cc()
        if max_cc == 0:
            # No build info — use old heuristic
            mock_props.major = 12  # Blackwell
            with patch("torch.cuda.get_device_properties", return_value=mock_props):
                self.assertTrue(_compiled_unsupported(cuda_device))

            mock_props.major = 9  # Hopper
            with patch("torch.cuda.get_device_properties", return_value=mock_props):
                self.assertFalse(_compiled_unsupported(cuda_device))
        else:
            # With build info: device_cc > max_cc means unsupported
            mock_props.major = max_cc // 100
            mock_props.minor = max_cc % 100
            with patch("torch.cuda.get_device_properties", return_value=mock_props):
                self.assertFalse(_compiled_unsupported(cuda_device))

            mock_props.major = max_cc // 100 + 2  # beyond compiled range
            with patch("torch.cuda.get_device_properties", return_value=mock_props):
                self.assertTrue(_compiled_unsupported(cuda_device))

    def test_compiled_unsupported_logic(self):
        """Test that unsupported devices are correctly detected."""
        cpu_device = torch.device("cpu")
        self.assertFalse(_compiled_unsupported(cpu_device))

        cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if cuda_device.type == "cuda":
            cc = torch.cuda.get_device_properties(cuda_device)
            device_cc = cc.major * 100 + cc.minor
            max_cc = _get_max_cc()
            if max_cc == 0:
                expected = cc.major >= 12
            else:
                expected = device_cc > max_cc
            self.assertEqual(_compiled_unsupported(cuda_device), expected)


if __name__ == "__main__":
    unittest.main()
