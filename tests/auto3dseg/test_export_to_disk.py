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

from monai.apps.auto3dseg.bundle_gen import BundleAlgo


class TestExportToDiskPathConfinement(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.templates = os.path.join(self.root, "templates")
        os.makedirs(os.path.join(self.templates, "net", "scripts"))
        with open(os.path.join(self.templates, "net", "scripts", "train.py"), "w") as f:
            f.write("# template\n")
        self.output = os.path.join(self.root, "work")
        os.makedirs(self.output)
        self.sibling = os.path.join(self.root, "sibling")
        os.makedirs(self.sibling)
        with open(os.path.join(self.sibling, "keep.txt"), "w") as f:
            f.write("keep\n")

    def tearDown(self):
        self._tmp.cleanup()

    def _algo(self, name: str = "net") -> BundleAlgo:
        algo = BundleAlgo(template_path=self.templates)
        algo.name = name
        return algo

    def _assert_sibling_untouched(self):
        self.assertEqual(os.listdir(self.sibling), ["keep.txt"])

    def test_valid_name_is_exported(self):
        algo = self._algo()
        algo.export_to_disk(self.output, "net_0", fill_template=False)
        self.assertEqual(algo.output_path, os.path.join(self.output, "net_0"))
        self.assertTrue(os.path.isfile(os.path.join(self.output, "net_0", "scripts", "train.py")))

    def test_re_export_replaces_existing_directory(self):
        algo = self._algo()
        algo.export_to_disk(self.output, "net_0", fill_template=False)
        stale = os.path.join(self.output, "net_0", "stale.txt")
        open(stale, "w").close()
        algo.export_to_disk(self.output, "net_0", fill_template=False)
        self.assertFalse(os.path.exists(stale))

    def test_algo_name_outside_output_path_is_rejected(self):
        for algo_name in (
            "../sibling",
            os.path.join("..", "sibling"),
            self.sibling,
            "a/b",
            ".",
            "..",
            "",
            "C:sibling",
            "a\x00b",
        ):
            with self.subTest(algo_name=algo_name):
                with self.assertRaises(ValueError):
                    self._algo().export_to_disk(self.output, algo_name, fill_template=False)
                self._assert_sibling_untouched()

    def test_symlinked_destination_outside_output_path_is_rejected(self):
        try:
            os.symlink(self.sibling, os.path.join(self.output, "linked"), target_is_directory=True)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"cannot create symlinks here: {e}")
        with self.assertRaisesRegex(ValueError, "not inside"):
            self._algo().export_to_disk(self.output, "linked", fill_template=False)
        self._assert_sibling_untouched()

    def test_invalid_source_keeps_existing_export(self):
        """An Algo name that cannot be copied from is rejected before the existing export is replaced."""
        self._algo().export_to_disk(self.output, "net_0", fill_template=False)
        for name in ("a\x00b", "missing"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self._algo(name=name).export_to_disk(self.output, "net_0", fill_template=False)
                self.assertTrue(os.path.isfile(os.path.join(self.output, "net_0", "scripts", "train.py")))

    def test_algo_name_attribute_outside_template_path_is_rejected(self):
        with self.assertRaises(ValueError):
            self._algo(name="../sibling").export_to_disk(self.output, "net_0", fill_template=False)
        self.assertFalse(os.path.exists(os.path.join(self.output, "net_0")))


if __name__ == "__main__":
    unittest.main()
