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
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from monai.auto3dseg.utils import (
    _add_path_with_parent,
    _make_json_serializable,
    algo_from_json,
    algo_from_pickle,
    algo_to_pickle,
)


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
        assert _make_json_serializable(Path("/some/path")) == os.fspath(Path("/some/path"))

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


class TestPickleGate(unittest.TestCase):
    """Pickle (de)serialization is gated behind MONAI_ALLOW_PICKLE=1."""

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        os.environ.pop("MONAI_ALLOW_PICKLE", None)
        self.addCleanup(patcher.stop)

    def test_algo_to_pickle_disabled_by_default(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "MONAI_ALLOW_PICKLE"):
            algo_to_pickle(object())  # type: ignore[arg-type]

    def test_algo_from_pickle_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = os.path.join(tmpdir, "algo_object.pkl")
            with open(pkl, "wb") as f:
                f.write(b"not used")
            with self.assertRaisesRegex(RuntimeError, "MONAI_ALLOW_PICKLE"):
                algo_from_pickle(pkl)

    def test_algo_from_json_legacy_pkl_disabled_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl = os.path.join(tmpdir, "algo_object.pkl")
            with open(pkl, "wb") as f:
                pickle.dump({"algo_bytes": b"x", "template_path": None}, f)
            with self.assertRaisesRegex(RuntimeError, "MONAI_ALLOW_PICKLE"):
                algo_from_json(pkl)


if __name__ == "__main__":
    unittest.main()
