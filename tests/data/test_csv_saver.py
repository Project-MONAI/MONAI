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

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from monai.data import CSVSaver


class TestCSVSaver(unittest.TestCase):

    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:
            saver = CSVSaver(output_dir=tempdir, filename="predictions.csv", delimiter="\t")
            meta_data = {"filename_or_obj": ["testfile" + str(i) for i in range(8)]}
            saver.save_batch(torch.zeros(8), meta_data)
            saver.finalize()
            filepath = os.path.join(tempdir, "predictions.csv")
            self.assertTrue(os.path.exists(filepath))
            with open(filepath) as f:
                reader = csv.reader(f, delimiter="\t")
                i = 0
                for row in reader:
                    self.assertEqual(row[0], "testfile" + str(i))
                    self.assertEqual(np.array(row[1:]).astype(np.float32), 0.0)
                    i += 1
                self.assertEqual(i, 8)

    def test_utf8_non_ascii_filename_and_metadata(self):
        patient_name = "\u60a3\u8005_01_\U0001f600.nii.gz"
        csv_filename = "\u9884\u6d4b_\U0001f600.csv"
        with tempfile.TemporaryDirectory() as tempdir:
            saver = CSVSaver(output_dir=tempdir, filename=csv_filename, delimiter="\t")
            saver.save_batch(torch.tensor([1, 2]), {"filename_or_obj": [patient_name, "patient_02"]})
            saver.finalize()

            filepath = os.path.join(tempdir, csv_filename)
            with open(filepath, encoding="utf-8") as f:
                rows = [line.split("\t") for line in f.read().splitlines()]

        self.assertEqual(rows[0], [patient_name, "1.0"])
        self.assertEqual(rows[1], ["patient_02", "2.0"])

    def test_utf8_append_preserves_existing_rows(self):
        patient_name = "\u60a3\u8005_02_\u75c5\u7076.nii.gz"
        csv_filename = "predictions.csv"
        with tempfile.TemporaryDirectory() as tempdir:
            saver = CSVSaver(output_dir=tempdir, filename=csv_filename, delimiter="\t")
            saver.save(torch.tensor(1), {"filename_or_obj": patient_name})
            saver.finalize()

            appender = CSVSaver(output_dir=tempdir, filename=csv_filename, overwrite=False, delimiter="\t")
            appender.save(torch.tensor(2), {"filename_or_obj": "patient_03"})
            appender.finalize()

            filepath = os.path.join(tempdir, csv_filename)
            with open(filepath, encoding="utf-8") as f:
                rows = [line.split("\t") for line in f.read().splitlines()]

        self.assertEqual(rows[0], [patient_name, "1.0"])
        self.assertEqual(rows[1], ["patient_03", "2.0"])

    def test_utf8_append_rejects_non_utf8_existing_file(self):
        csv_filename = "predictions.csv"
        legacy_contents = b"caf\xe9,1.0\n"
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir, csv_filename)
            with open(filepath, "wb") as f:
                f.write(legacy_contents)

            appender = CSVSaver(output_dir=tempdir, filename=csv_filename, overwrite=False, delimiter="\t")
            appender.save(torch.tensor(2), {"filename_or_obj": "patient_03"})
            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                appender.finalize()

            with open(filepath, "rb") as f:
                self.assertEqual(f.read(), legacy_contents)

    def test_utf8_flush_rejects_non_utf8_suffix(self):
        csv_filename = "predictions.csv"
        patient_name = "\u60a3\u8005_04_\u75c5\u7076.nii.gz"
        with tempfile.TemporaryDirectory() as tempdir:
            saver = CSVSaver(output_dir=tempdir, filename=csv_filename, flush=True, delimiter="\t")
            saver.save_batch(torch.tensor([1]), {"filename_or_obj": [patient_name]})
            filepath = os.path.join(tempdir, csv_filename)
            with open(filepath, "rb") as f:
                valid_contents = f.read()
            with open(filepath, "ab") as f:
                f.write(b"\xff")

            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                saver.save_batch(torch.tensor([2]), {"filename_or_obj": ["patient_05"]})

            with open(filepath, "rb") as f:
                self.assertEqual(f.read(), valid_contents + b"\xff")

    def test_utf8_flush_rejects_same_size_rewrite(self):
        csv_filename = "predictions.csv"
        with tempfile.TemporaryDirectory() as tempdir:
            saver = CSVSaver(output_dir=tempdir, filename=csv_filename, flush=True, delimiter="\t")
            saver.save_batch(torch.tensor([1]), {"filename_or_obj": ["patient_06"]})
            filepath = os.path.join(tempdir, csv_filename)
            with open(filepath, "rb") as f:
                valid_contents = f.read()
            with open(filepath, "wb") as f:
                f.write(b"\xff" + valid_contents[1:])
            original_stat = os.stat(filepath)
            os.utime(filepath, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000))

            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                saver.save_batch(torch.tensor([2]), {"filename_or_obj": ["patient_07"]})

            with open(filepath, "rb") as f:
                self.assertEqual(f.read(), b"\xff" + valid_contents[1:])

    def test_finalize_does_not_replay_rows_when_stat_fails_after_write(self):
        """A failed fingerprint stat must not leave already-written rows cached."""
        csv_filename = "predictions.csv"
        with tempfile.TemporaryDirectory() as tempdir:
            saver = CSVSaver(output_dir=tempdir, filename=csv_filename, delimiter="\t")
            saver.save(torch.tensor(1), {"filename_or_obj": "patient_01"})
            filepath = Path(tempdir) / csv_filename
            original_stat = Path.stat

            def stat_with_failure(path, *args, **kwargs):
                """Fail only when the target file exists, after its contents are written."""
                if path == filepath and os.path.exists(filepath):
                    raise OSError("stat failed")
                return original_stat(path, *args, **kwargs)

            with mock.patch.object(Path, "stat", new=stat_with_failure):
                with self.assertRaisesRegex(OSError, "stat failed"):
                    saver.finalize()

            self.assertFalse(saver.get_cache())
            with open(filepath, "rb") as f:
                written_contents = f.read()

            saver.finalize()

            with open(filepath, "rb") as f:
                self.assertEqual(f.read(), written_contents)


if __name__ == "__main__":
    unittest.main()
