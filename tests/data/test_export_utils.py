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

import torch

from monai.config import get_config_values
from monai.data import load_exported_program, save_exported_program
from monai.data.torchscript_utils import METADATA_FILENAME
from monai.utils import JITMetadataKeys


class TestModule(torch.nn.Module):
    __test__ = False

    def forward(self, x):
        return x + 10


class TestExportUtils(unittest.TestCase):
    def test_save_exported_program(self):
        """Save an exported program without metadata to a file."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))
        with tempfile.TemporaryDirectory() as tempdir:
            save_exported_program(ep, f"{tempdir}/test")
            self.assertTrue(os.path.isfile(f"{tempdir}/test.pt2"))

    def test_save_exported_program_ext(self):
        """Save an exported program to a file with custom extension."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))
        with tempfile.TemporaryDirectory() as tempdir:
            save_exported_program(ep, f"{tempdir}/test.zip")
            self.assertTrue(os.path.isfile(f"{tempdir}/test.zip"))

    def test_save_with_metadata(self):
        """Save an exported program with metadata to a file."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))
        test_metadata = {"foo": [1, 2], "bar": "string"}

        with tempfile.TemporaryDirectory() as tempdir:
            save_exported_program(ep, f"{tempdir}/test", meta_values=test_metadata)
            self.assertTrue(os.path.isfile(f"{tempdir}/test.pt2"))

    def test_load_exported_program(self):
        """Save then load an exported program with no extra metadata."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))

        with tempfile.TemporaryDirectory() as tempdir:
            save_exported_program(ep, f"{tempdir}/test")
            loaded_ep, meta, extra_files = load_exported_program(f"{tempdir}/test.pt2")

        del meta[JITMetadataKeys.TIMESTAMP.value]
        self.assertEqual(meta, get_config_values())
        self.assertEqual(extra_files, {})

        # Verify the loaded program produces the same output
        result = loaded_ep.module()(torch.tensor(5.0))
        self.assertEqual(result.item(), 15.0)

    def test_load_with_metadata(self):
        """Save then load an exported program with metadata."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))
        test_metadata = {"foo": [1, 2], "bar": "string"}

        with tempfile.TemporaryDirectory() as tempdir:
            save_exported_program(ep, f"{tempdir}/test", meta_values=test_metadata)
            _, meta, extra_files = load_exported_program(f"{tempdir}/test.pt2")

        del meta[JITMetadataKeys.TIMESTAMP.value]

        test_compare = get_config_values()
        test_compare.update(test_metadata)
        self.assertEqual(meta, test_compare)
        self.assertEqual(extra_files, {})

    def test_save_load_more_extra_files(self):
        """Save then load extra file data from an exported program."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))
        test_metadata = {"foo": [1, 2], "bar": "string"}
        more_extra_files = {"test.txt": "This is test data"}

        with tempfile.TemporaryDirectory() as tempdir:
            save_exported_program(ep, f"{tempdir}/test", meta_values=test_metadata, more_extra_files=more_extra_files)
            self.assertTrue(os.path.isfile(f"{tempdir}/test.pt2"))

            _, _, loaded_extra_files = load_exported_program(f"{tempdir}/test.pt2", more_extra_files=("test.txt",))
            self.assertEqual(more_extra_files["test.txt"], loaded_extra_files["test.txt"])

    def test_load_without_metadata(self):
        """Load a file saved directly with torch.export.save, which has no metadata."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))

        with tempfile.TemporaryDirectory() as tempdir:
            torch.export.save(ep, f"{tempdir}/test.pt2")
            _, meta, extra_files = load_exported_program(f"{tempdir}/test.pt2")

        self.assertEqual(meta, {})
        self.assertEqual(extra_files, {})

    def test_reserved_extra_file_name(self):
        """The metadata filename cannot be used for extra files."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))

        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaisesRegex(ValueError, METADATA_FILENAME):
                save_exported_program(ep, f"{tempdir}/test", more_extra_files={METADATA_FILENAME: "{}"})

    def test_bytes_extra_file(self):
        """Extra files given as UTF-8 bytes load back as strings."""
        ep = torch.export.export(TestModule(), args=(torch.tensor(1.0),))

        with tempfile.TemporaryDirectory() as tempdir:
            save_exported_program(ep, f"{tempdir}/test", more_extra_files={"test.json": b"{}"})
            _, _, extra_files = load_exported_program(f"{tempdir}/test.pt2", more_extra_files=("test.json",))

        self.assertEqual(extra_files["test.json"], "{}")


if __name__ == "__main__":
    unittest.main()
