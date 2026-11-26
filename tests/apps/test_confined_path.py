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

from monai.apps.utils import confined_path


class TestConfinedPath(unittest.TestCase):
    def test_plain_child(self):
        with tempfile.TemporaryDirectory() as root:
            got = confined_path(root, "model.pt")
            self.assertEqual(got, Path(root).resolve() / "model.pt")

    def test_nested_child_allowed_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            got = confined_path(root, "sub/dir/model.pt")
            self.assertEqual(got, Path(root).resolve() / "sub" / "dir" / "model.pt")

    def test_parent_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for bad in ["../x", "../../x", "a/../../x", "sub/../../x"]:
                with self.assertRaises(ValueError):
                    confined_path(root, bad)

    def test_absolute_escape_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                confined_path(root, "/etc/passwd")

    def test_absolute_inside_root_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            inside = os.path.join(Path(root).resolve(), "a", "b")
            got = confined_path(root, inside)
            self.assertEqual(got, Path(inside))

    def test_sibling_prefix_not_inside(self):
        # /x/foo2 must not be treated as inside /x/foo
        with tempfile.TemporaryDirectory() as base:
            root = os.path.join(base, "foo")
            os.makedirs(root)
            os.makedirs(os.path.join(base, "foo2"))
            with self.assertRaises(ValueError):
                confined_path(root, "../foo2/x")

    def test_symlinked_base_directory(self):
        # a symlinked root (mirrors /tmp -> /private/tmp on macOS) must not cause a false rejection
        with tempfile.TemporaryDirectory() as base:
            real = os.path.join(base, "real")
            os.makedirs(real)
            link = os.path.join(base, "link")
            try:
                os.symlink(real, link)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks not supported on this platform")
            got = confined_path(link, "model.pt")
            self.assertEqual(got, Path(real).resolve() / "model.pt")

    def test_root_itself_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            for same in [".", ""]:
                with self.assertRaises(ValueError):
                    confined_path(root, same)

    def test_require_existing_dir(self):
        with tempfile.TemporaryDirectory() as base:
            missing = os.path.join(base, "nope")
            with self.assertRaises(ValueError):
                confined_path(missing, "x", require_existing_dir=True)
            # without the flag, a non-existent root is fine
            self.assertEqual(confined_path(missing, "x"), Path(missing).resolve() / "x")

    def test_allow_subdirs_false_rejects_nested(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(confined_path(root, "model.pt", allow_subdirs=False), Path(root).resolve() / "model.pt")
            with self.assertRaises(ValueError):
                confined_path(root, "sub/model.pt", allow_subdirs=False)

    def test_create_parents(self):
        with tempfile.TemporaryDirectory() as root:
            got = confined_path(root, "a/b/model.pt", create_parents=True)
            self.assertTrue(got.parent.is_dir())


if __name__ == "__main__":
    unittest.main()
