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

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from monai.auto3dseg.utils import _add_path_with_parent, _make_json_serializable


class TestMakeJsonSerializable(unittest.TestCase):
    def test_primitives(self) -> None:
        assert _make_json_serializable(None) is None
        assert _make_json_serializable("hello") == "hello"
        assert _make_json_serializable(42) == 42
        assert _make_json_serializable(3.14) == 3.14
        assert _make_json_serializable(True) is True

    def test_collections(self) -> None:
        assert _make_json_serializable([1, 2, 3]) == [1, 2, 3]
        assert _make_json_serializable((1, 2)) == [1, 2]
        assert _make_json_serializable({"a": 1}) == {"a": 1}

    def test_numpy(self) -> None:
        arr = np.array([1, 2, 3])
        assert _make_json_serializable(arr) == [1, 2, 3]
        assert _make_json_serializable(np.int64(5)) == 5
        assert _make_json_serializable(np.float32(2.5)) == 2.5

    def test_torch_tensor(self) -> None:
        t = torch.tensor([1.0, 2.0])
        result = _make_json_serializable(t)
        assert result == [1.0, 2.0]

    def test_path(self) -> None:
        p = Path("/some/path")
        # Use str(p) since path separators differ on Windows vs Unix
        assert _make_json_serializable(p) == str(p)

    def test_fallback(self) -> None:
        class Custom:
            def __str__(self) -> str:
                return "custom"

        assert _make_json_serializable(Custom()) == "custom"


class TestAddPathWithParent(unittest.TestCase):
    def test_valid_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: list[str] = []
            _add_path_with_parent(paths, tmpdir)
            assert len(paths) == 2
            assert os.path.abspath(tmpdir) in paths
            assert os.path.abspath(os.path.join(tmpdir, "..")) in paths

    def test_none_path(self) -> None:
        paths: list[str] = []
        _add_path_with_parent(paths, None)
        assert len(paths) == 0

    def test_nonexistent_path(self) -> None:
        paths: list[str] = []
        _add_path_with_parent(paths, "/nonexistent/path/12345")
        assert len(paths) == 0


if __name__ == "__main__":
    unittest.main()
