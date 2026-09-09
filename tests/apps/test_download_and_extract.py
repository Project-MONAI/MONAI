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

import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from parameterized import parameterized

from monai.apps import download_and_extract, download_url, extractall
from monai.apps.utils import HashCheckError
from tests.test_utils import SkipIfNoModule, skip_if_downloading_fails, skip_if_quick, testing_data_config


@SkipIfNoModule("requests")
class TestDownloadAndExtract(unittest.TestCase):
    def setUp(self):
        self.testing_dir = Path(__file__).parents[1] / "testing_data"
        self.config = testing_data_config("images", "mednist")
        self.url = self.config["url"]
        self.hash_val = self.config["hash_val"]
        self.hash_type = self.config["hash_type"]

    @skip_if_quick
    def test_download_and_extract_success(self):
        """End-to-end: download and extract should succeed with correct hash."""
        filepath = self.testing_dir / "MedNIST.tar.gz"
        output_dir = self.testing_dir

        with skip_if_downloading_fails():
            download_and_extract(self.url, filepath, output_dir, hash_val=self.hash_val, hash_type=self.hash_type)

        self.assertTrue(filepath.exists(), "Downloaded file does not exist")
        self.assertTrue(any(output_dir.iterdir()), "Extraction output is empty")

    @skip_if_quick
    def test_download_url_hash_mismatch(self):
        """download_url should raise HashCheckError on hash mismatch."""
        filepath = self.testing_dir / "MedNIST.tar.gz"

        with skip_if_downloading_fails():
            # First ensure file is downloaded correctly
            download_url(self.url, filepath, hash_val=self.hash_val, hash_type=self.hash_type)

        # Now test incorrect hash
        with self.assertRaises(HashCheckError):
            download_url(self.url, filepath, hash_val="0" * len(self.hash_val), hash_type=self.hash_type)

    @skip_if_quick
    def test_extractall_hash_mismatch(self):
        """extractall should raise HashCheckError when hash is incorrect."""
        filepath = self.testing_dir / "MedNIST.tar.gz"
        output_dir = self.testing_dir

        with skip_if_downloading_fails():
            download_url(self.url, filepath, hash_val=self.hash_val, hash_type=self.hash_type)

        with self.assertRaises(HashCheckError):
            extractall(filepath, output_dir, hash_val="0" * len(self.hash_val), hash_type=self.hash_type)

    @skip_if_quick
    @parameterized.expand([("icon", "tar"), ("favicon", "zip")])
    def test_download_and_extract_various_formats(self, key, file_type):
        """Verify different archive formats download and extract correctly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_spec = testing_data_config("images", key)

            with skip_if_downloading_fails():
                download_and_extract(
                    img_spec["url"],
                    output_dir=tmp_dir,
                    hash_val=img_spec["hash_val"],
                    hash_type=img_spec["hash_type"],
                    file_type=file_type,
                )

            self.assertTrue(any(Path(tmp_dir).iterdir()), f"Extraction failed for format: {file_type}")


class TestPathTraversalProtection(unittest.TestCase):
    """Test cases for path traversal attack protection in extractall function."""

    def test_valid_zip_extraction(self):
        """Test that valid zip files extract successfully without raising exceptions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid zip file
            zip_path = Path(tmp_dir) / "valid_test.zip"
            extract_dir = Path(tmp_dir) / "extract"
            extract_dir.mkdir()

            # Create zip with normal file structure
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("normal_file.txt", "This is a normal file")
                zf.writestr("subdir/nested_file.txt", "This is a nested file")
                zf.writestr("another_file.json", '{"key": "value"}')

            # This should not raise any exception
            try:
                extractall(str(zip_path), str(extract_dir))

                # Verify files were extracted correctly
                self.assertTrue((extract_dir / "normal_file.txt").exists())
                self.assertTrue((extract_dir / "subdir" / "nested_file.txt").exists())
                self.assertTrue((extract_dir / "another_file.json").exists())

                # Verify content
                with open(extract_dir / "normal_file.txt") as f:
                    self.assertEqual(f.read(), "This is a normal file")

            except Exception as e:
                self.fail(f"Valid zip extraction should not raise exception: {e}")

    def test_malicious_zip_path_traversal(self):
        """Test that malicious zip files with path traversal attempts raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create malicious zip file with path traversal
            zip_path = Path(tmp_dir) / "malicious_test.zip"
            extract_dir = Path(tmp_dir) / "extract"
            extract_dir.mkdir()

            # Create zip with malicious paths
            with zipfile.ZipFile(zip_path, "w") as zf:
                # Try to write outside extraction directory
                zf.writestr("../../../etc/malicious.txt", "malicious content")
                zf.writestr("normal_file.txt", "normal content")

            # This should raise ValueError due to path traversal detection
            with self.assertRaises(ValueError) as context:
                extractall(str(zip_path), str(extract_dir))

            self.assertIn("unsafe path", str(context.exception).lower())

    def test_valid_tar_extraction(self):
        """Test that valid tar files extract successfully without raising exceptions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a valid tar file
            tar_path = Path(tmp_dir) / "valid_test.tar.gz"
            extract_dir = Path(tmp_dir) / "extract"
            extract_dir.mkdir()

            # Create tar with normal file structure
            with tarfile.open(tar_path, "w:gz") as tf:
                # Create temporary files to add to tar
                temp_file1 = Path(tmp_dir) / "temp1.txt"
                temp_file1.write_text("This is a normal file")
                tf.add(temp_file1, arcname="normal_file.txt")

                temp_file2 = Path(tmp_dir) / "temp2.txt"
                temp_file2.write_text("This is a nested file")
                tf.add(temp_file2, arcname="subdir/nested_file.txt")

            # This should not raise any exception
            try:
                extractall(str(tar_path), str(extract_dir))

                # Verify files were extracted correctly
                self.assertTrue((extract_dir / "normal_file.txt").exists())
                self.assertTrue((extract_dir / "subdir" / "nested_file.txt").exists())

                # Verify content
                with open(extract_dir / "normal_file.txt") as f:
                    self.assertEqual(f.read(), "This is a normal file")

            except Exception as e:
                self.fail(f"Valid tar extraction should not raise exception: {e}")

    def test_malicious_tar_path_traversal(self):
        """Test that malicious tar files with path traversal attempts raise ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create malicious tar file with path traversal
            tar_path = Path(tmp_dir) / "malicious_test.tar.gz"
            extract_dir = Path(tmp_dir) / "extract"
            extract_dir.mkdir()

            # Create tar with malicious paths
            with tarfile.open(tar_path, "w:gz") as tf:
                # Create a temporary file
                temp_file = Path(tmp_dir) / "temp.txt"
                temp_file.write_text("malicious content")

                # Add with malicious path (trying to write outside extraction directory)
                tf.add(temp_file, arcname="../../../etc/malicious.txt")

            # This should raise ValueError due to path traversal detection
            with self.assertRaises(ValueError) as context:
                extractall(str(tar_path), str(extract_dir))

            self.assertIn("unsafe path", str(context.exception).lower())

    def test_absolute_path_protection(self):
        """Test protection against absolute paths in archives."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create zip with absolute path
            zip_path = Path(tmp_dir) / "absolute_path_test.zip"
            extract_dir = Path(tmp_dir) / "extract"
            extract_dir.mkdir()

            with zipfile.ZipFile(zip_path, "w") as zf:
                # Try to use absolute path
                zf.writestr("/etc/passwd_bad", "malicious content")

            # This should raise ValueError due to absolute path detection
            with self.assertRaises(ValueError) as context:
                extractall(str(zip_path), str(extract_dir))

            self.assertIn("unsafe path", str(context.exception).lower())

    def test_malicious_symlink_protection(self):
        """Test protection against malicious symlinks in tar archives."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create malicious tar file with symlink
            tar_path = Path(tmp_dir) / "malicious_symlink_test.tar.gz"
            extract_dir = Path(tmp_dir) / "extract"
            extract_dir.mkdir()

            # Create tar with malicious symlink
            with tarfile.open(tar_path, "w:gz") as tf:
                temp_file = Path(tmp_dir) / "normal.txt"
                temp_file.write_text("normal content")
                tf.add(temp_file, arcname="normal.txt")

                symlink_info = tarfile.TarInfo(name="malicious_symlink.txt")
                symlink_info.type = tarfile.SYMTYPE
                symlink_info.linkname = "../../../etc/passwd_bad"
                symlink_info.size = 0
                tf.addfile(symlink_info)

            with self.assertRaises(ValueError) as context:
                extractall(str(tar_path), str(extract_dir))

            error_msg = str(context.exception).lower()
            self.assertTrue("unsafe path" in error_msg or "symlink" in error_msg)

    def test_malicious_hardlink_protection(self):
        """Test protection against malicious hard links in tar archives."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create malicious tar file with hard link
            tar_path = Path(tmp_dir) / "malicious_hardlink_test.tar.gz"
            extract_dir = Path(tmp_dir) / "extract"
            extract_dir.mkdir()

            # Create tar with malicious hard link
            with tarfile.open(tar_path, "w:gz") as tf:
                temp_file = Path(tmp_dir) / "normal.txt"
                temp_file.write_text("normal content")
                tf.add(temp_file, arcname="normal.txt")

                hardlink_info = tarfile.TarInfo(name="malicious_hardlink.txt")
                hardlink_info.type = tarfile.LNKTYPE
                hardlink_info.linkname = "/etc/passwd_bad"
                hardlink_info.size = 0
                tf.addfile(hardlink_info)

            with self.assertRaises(ValueError) as context:
                extractall(str(tar_path), str(extract_dir))

            error_msg = str(context.exception).lower()
            self.assertTrue("unsafe path" in error_msg or "hardlink" in error_msg)


if __name__ == "__main__":
    unittest.main()
