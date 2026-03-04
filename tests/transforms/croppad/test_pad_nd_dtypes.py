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
"""
Tests for pad_nd dtype support and backend selection.
Validates PyTorch padding preference and NumPy fallback behavior.
"""
from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch
from parameterized.parameterized import parameterized

import monai.transforms.croppad.functional as F
from monai.transforms.croppad.functional import pad_nd

DTYPES = [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.float32]
MODES_DTYPES = [
    ("constant", torch.bool),
    ("constant", torch.int8),
    ("constant", torch.float32),
    ("reflect", torch.bool),
    ("reflect", torch.int8),
    ("reflect", torch.float32),
    ("replicate", torch.bool),
    ("replicate", torch.int8),
    ("replicate", torch.float32),
]


class TestPadNdDtypes(unittest.TestCase):
    def test_pad_uses_pt_for_bool(self):
        """Test that pad_nd uses PyTorch backend for bool dtype in constant mode."""
        img = torch.ones((1, 4, 4), dtype=torch.bool)
        to_pad = [(0, 0), (1, 1), (2, 2)]
        with (
            patch.object(F, "_pt_pad", wraps=F._pt_pad) as mock_pt,
            patch.object(F, "_np_pad", wraps=F._np_pad) as mock_np,
        ):
            out = pad_nd(img, to_pad, mode="constant", value=0)

        self.assertTrue(mock_pt.called)
        self.assertFalse(mock_np.called)
        self.assertEqual(out.dtype, img.dtype)
        self.assertEqual(out.shape, (1, 6, 8))

    def test_pad_falls_back_to_np_if_pt_raises(self):
        """Test that pad_nd falls back to NumPy when PyTorch raises NotImplementedError."""
        img = torch.ones((1, 4, 4), dtype=torch.bool)
        to_pad = [(0, 0), (1, 1), (2, 2)]
        with (
            patch.object(F, "_pt_pad", new=Mock(side_effect=NotImplementedError("no"))) as mock_pt,
            patch.object(F, "_np_pad", wraps=F._np_pad) as mock_np,
        ):
            out = pad_nd(img, to_pad, mode="constant", value=0)

        self.assertTrue(mock_pt.called)
        self.assertTrue(mock_np.called)
        self.assertEqual(out.dtype, img.dtype)
        self.assertEqual(out.shape, (1, 6, 8))

    @parameterized.expand(DTYPES)
    def test_pad_dtype_no_error_and_dtype_preserved(self, dtype):
        """Test that pad_nd handles various dtypes without error and preserves dtype.
        Args:
            dtype: Input dtype under test.
        """
        img = torch.ones((1, 4, 4), dtype=dtype)
        to_pad = [(0, 0), (1, 1), (2, 2)]
        out = pad_nd(img, to_pad, mode="constant", value=0)

        self.assertEqual(out.shape, (1, 6, 8))
        self.assertEqual(out.dtype, img.dtype)

    @parameterized.expand(MODES_DTYPES)
    def test_pad_multiple_modes_dtype_preserved(self, mode, dtype):
        """Test that pad_nd preserves dtype across multiple padding modes.
        Args:
            mode: Padding mode under test.
            dtype: Input dtype under test.
        """
        img = torch.ones((1, 4, 4), dtype=dtype)
        to_pad = [(0, 0), (1, 1), (2, 2)]

        kwargs = {"value": 0} if mode == "constant" else {}
        out = pad_nd(img, to_pad, mode=mode, **kwargs)

        self.assertEqual(out.shape, (1, 6, 8))
        self.assertEqual(out.dtype, img.dtype)

    def test_value_with_non_constant_mode_raises(self):
        """Test that pad_nd raises ValueError when 'value' is provided with non-constant mode."""
        img = torch.ones((1, 4, 4))
        to_pad = [(0, 0), (1, 1), (2, 2)]
        with self.assertRaises(ValueError):
            pad_nd(img, to_pad, mode="reflect", value=0)


if __name__ == "__main__":
    unittest.main()
