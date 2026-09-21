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

import hashlib
import os
import shutil
import tempfile
import unittest
import warnings
import zipfile
from unittest.mock import patch

import numpy as np
from parameterized import parameterized

from monai.apps import check_hash, download_and_extract, download_url, extractall

TEST_CASE_1 = ["b94716452086a054208395e8c9d1ae2a", "md5", True]

TEST_CASE_2 = ["abcdefg", "md5", False]

TEST_CASE_3 = [None, "md5", True]

TEST_CASE_4 = [None, "sha1", True]

TEST_CASE_5 = ["b4dc3c246b298eae37cefdfdd2a50b091ffd5e69", "sha1", True]


class TestCheckMD5(unittest.TestCase):
    @staticmethod
    def _hash_file(filename, hash_type):
        hash_func = getattr(hashlib, hash_type)
        with open(filename, "rb") as f:
            return hash_func(f.read()).hexdigest()

    @staticmethod
    def _write_file(filename, content=b"monai hash fixture"):
        with open(filename, "wb") as f:
            f.write(content)

    @classmethod
    def _download_side_effect(cls, source):
        def _copy(_url, destination, progress=True):
            del progress
            shutil.copyfile(source, destination)

        return _copy

    @classmethod
    def _create_zip_fixture(cls, tempdir):
        archive = os.path.join(tempdir, "fixture.zip")
        with zipfile.ZipFile(archive, "w") as zip_file:
            zip_file.writestr("fixture/data.txt", "monai")
        return archive

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4, TEST_CASE_5])
    def test_result(self, md5_value, t, expected_result):
        test_image = np.ones((5, 5, 3))
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_file.png")
            test_image.tofile(filename)

            result = check_hash(filename, md5_value, hash_type=t)
            self.assertTrue(result == expected_result)

    def test_hash_type_error(self):
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as tempdir:
                check_hash(tempdir, "test_hash", "test_type")

    def test_warns_when_val_is_none(self):
        test_image = np.ones((5, 5, 3))
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_file.png")
            test_image.tofile(filename)
            with self.assertWarns(UserWarning):
                result = check_hash(filename, None, hash_type="sha256")
            self.assertTrue(result)

    def test_default_hash_type_is_sha256(self):
        test_image = np.ones((5, 5, 3))
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_file.png")
            test_image.tofile(filename)
            sha256 = hashlib.sha256(test_image.tobytes()).hexdigest()
            self.assertTrue(check_hash(filename, sha256, hash_type="sha256"))

    def test_omitting_hash_type_emits_future_warning(self):
        def assert_single_future_warning(callable_obj, *args, **kwargs):
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                callable_obj(*args, **kwargs)

            future_warnings = [w for w in recorded if issubclass(w.category, FutureWarning)]
            self.assertEqual(len(future_warnings), 1)
            self.assertIn('hash_type="md5"', str(future_warnings[0].message))
            self.assertIn('hash_type="sha256"', str(future_warnings[0].message))

        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "fixture.bin")
            self._write_file(filename)
            sha256 = self._hash_file(filename, "sha256")

            download_target = os.path.join(tempdir, "downloaded.bin")
            archive = self._create_zip_fixture(tempdir)
            archive_sha256 = self._hash_file(archive, "sha256")

            assert_single_future_warning(check_hash, filename, sha256)
            with patch("monai.apps.utils._download_with_progress", side_effect=self._download_side_effect(filename)):
                assert_single_future_warning(
                    download_url, "https://example.com/fixture.bin", download_target, hash_val=sha256, progress=False
                )
            assert_single_future_warning(extractall, archive, os.path.join(tempdir, "extract"), hash_val=archive_sha256)
            with (
                patch("monai.apps.utils._download_with_progress", side_effect=self._download_side_effect(archive)),
                patch("monai.apps.utils.get_filename_from_url", return_value="fixture.zip"),
            ):
                assert_single_future_warning(
                    download_and_extract,
                    "https://example.com/fixture.zip",
                    filepath=os.path.join(tempdir, "downloaded.zip"),
                    output_dir=os.path.join(tempdir, "download_and_extract"),
                    hash_val=archive_sha256,
                    progress=False,
                )

    def test_explicit_sha256_does_not_emit_default_change_warning(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "fixture.bin")
            self._write_file(filename)
            sha256 = self._hash_file(filename, "sha256")

            download_target = os.path.join(tempdir, "downloaded.bin")
            archive = self._create_zip_fixture(tempdir)
            archive_sha256 = self._hash_file(archive, "sha256")

            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                self.assertTrue(check_hash(filename, sha256, hash_type="sha256"))
                with patch(
                    "monai.apps.utils._download_with_progress", side_effect=self._download_side_effect(filename)
                ):
                    download_url(
                        "https://example.com/fixture.bin",
                        download_target,
                        hash_val=sha256,
                        hash_type="sha256",
                        progress=False,
                    )
                extractall(archive, os.path.join(tempdir, "extract"), hash_val=archive_sha256, hash_type="sha256")
                with (
                    patch("monai.apps.utils._download_with_progress", side_effect=self._download_side_effect(archive)),
                    patch("monai.apps.utils.get_filename_from_url", return_value="fixture.zip"),
                ):
                    download_and_extract(
                        "https://example.com/fixture.zip",
                        filepath=os.path.join(tempdir, "downloaded.zip"),
                        output_dir=os.path.join(tempdir, "download_and_extract"),
                        hash_val=archive_sha256,
                        hash_type="sha256",
                        progress=False,
                    )

            future_warnings = [w for w in recorded if issubclass(w.category, FutureWarning)]
            self.assertEqual(future_warnings, [])

    def test_explicit_md5_still_works(self):
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "fixture.bin")
            self._write_file(filename)
            md5 = self._hash_file(filename, "md5")

            download_target = os.path.join(tempdir, "downloaded.bin")
            archive = self._create_zip_fixture(tempdir)
            archive_md5 = self._hash_file(archive, "md5")

            self.assertTrue(check_hash(filename, md5, hash_type="md5"))
            with patch("monai.apps.utils._download_with_progress", side_effect=self._download_side_effect(filename)):
                download_url(
                    "https://example.com/fixture.bin", download_target, hash_val=md5, hash_type="md5", progress=False
                )
            extractall(archive, os.path.join(tempdir, "extract"), hash_val=archive_md5, hash_type="md5")
            with (
                patch("monai.apps.utils._download_with_progress", side_effect=self._download_side_effect(archive)),
                patch("monai.apps.utils.get_filename_from_url", return_value="fixture.zip"),
            ):
                download_and_extract(
                    "https://example.com/fixture.zip",
                    filepath=os.path.join(tempdir, "downloaded.zip"),
                    output_dir=os.path.join(tempdir, "download_and_extract"),
                    hash_val=archive_md5,
                    hash_type="md5",
                    progress=False,
                )


if __name__ == "__main__":
    unittest.main()
