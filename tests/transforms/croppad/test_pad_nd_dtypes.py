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
from unittest.mock import Mock, patch
import pytest
import torch
import monai.transforms.croppad.functional as F
from monai.transforms.croppad.functional import pad_nd

def test_pad_uses_pt_for_bool():
    """Test that pad_nd uses PyTorch backend for bool dtype in constant mode."""
    img = torch.ones((1, 4, 4), dtype=torch.bool)
    to_pad = [(0, 0), (1, 1), (2, 2)]
    with patch.object(F, "_pt_pad", wraps=F._pt_pad) as mock_pt, patch.object(F, "_np_pad", wraps=F._np_pad) as mock_np:
        out = pad_nd(img, to_pad, mode="constant", value=0)

    assert mock_pt.called
    assert not mock_np.called
    assert out.dtype == img.dtype

def test_pad_falls_back_to_np_if_pt_raises():
    """Test that pad_nd falls back to NumPy when PyTorch raises NotImplementedError."""
    img = torch.ones((1, 4, 4), dtype=torch.bool)
    to_pad = [(0, 0), (1, 1), (2, 2)]
    with (
        patch.object(F, "_pt_pad", new=Mock(side_effect=NotImplementedError("no"))) as mock_pt,
        patch.object(F, "_np_pad", wraps=F._np_pad) as mock_np,
    ):
        out = pad_nd(img, to_pad, mode="constant", value=0)

    assert mock_pt.called
    assert mock_np.called
    assert out.dtype == img.dtype

@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.float32]
)
def test_pad_dtype_no_error_and_dtype_preserved(dtype):
    """Test that pad_nd handles various dtypes without error and preserves dtype."""
    img = torch.ones((1, 4, 4), dtype=dtype)
    to_pad = [(0, 0), (1, 1), (2, 2)]
    out = pad_nd(img, to_pad, mode="constant", value=0)

    assert out.shape == (1, 6, 8)
    assert out.dtype == img.dtype

@pytest.mark.parametrize("mode", ["constant", "reflect", "replicate"])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int8, torch.float32])
def test_pad_multiple_modes_dtype_preserved(mode, dtype):
    """Test that pad_nd preserves dtype across multiple padding modes."""
    img = torch.ones((1, 4, 4), dtype=dtype)
    to_pad = [(0, 0), (1, 1), (2, 2)]
    
    kwargs = {"value": 0} if mode == "constant" else {}
    out = pad_nd(img, to_pad, mode=mode, **kwargs)

    assert out.shape == (1, 6, 8)
    assert out.dtype == img.dtype
