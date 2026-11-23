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
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from parameterized import parameterized

from monai.apps import utils
from tests.test_utils import SkipIfNoModule


def _head_response(filename: str) -> SimpleNamespace:
    return SimpleNamespace(headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@SkipIfNoModule("requests")
class TestDownloadRemoteFilename(unittest.TestCase):
    """``download_and_extract`` without ``filepath`` keeps the server-named file in its temporary directory."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name).resolve()
        # download_and_extract creates its TemporaryDirectory under tmp_base, so a file that stays
        # inside it is at tmp_base/<tmp_dir>/<name>, and "../name" would land directly in tmp_base
        self.tmp_base = self.root / "tmp_base"
        self.tmp_base.mkdir()
        self.output_dir = self.root / "extracted"
        self.downloads: list[Path] = []
        patcher = mock.patch.object(tempfile, "tempdir", str(self.tmp_base))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._dir.cleanup)

    def _fake_download(self, url, filepath, **kwargs):
        """Stand-in for ``download_url``: writes a small zip archive at the requested destination."""
        filepath = Path(filepath)
        self.downloads.append(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(filepath, "w") as zf:
            zf.writestr("archive/data.txt", "payload")

    def _run(self, remote_name: str, head: mock.Mock | None = None):
        head = head or mock.Mock(return_value=_head_response(remote_name))
        with (
            mock.patch.object(utils.requests, "head", head),
            mock.patch.object(utils, "download_url", side_effect=self._fake_download) as download,
        ):
            utils.download_and_extract("https://example.invalid/archive.zip", output_dir=self.output_dir)
        return download, head

    def _files_outside_tmp_dirs(self) -> list[Path]:
        """Every file under ``self.root`` that is neither extracted output nor inside a per-call temp directory."""
        found = []
        for dirpath, _, filenames in os.walk(self.root):
            for name in filenames:
                path = Path(dirpath, name)
                if self.output_dir in path.parents:
                    continue
                if path.parent.parent == self.tmp_base:  # tmp_base/<TemporaryDirectory>/<file>
                    continue
                found.append(path)
        return found

    @parameterized.expand(
        [
            ("parent", "../outside.zip"),
            ("nested_parent", "../../outside.zip"),
            ("windows_parent", "..\\outside.zip"),
            ("dir_component", "sub/archive.zip"),
            ("trailing_parent", "sub/.."),
        ]
    )
    def test_name_with_path_component_is_rejected(self, _, remote_name):
        with self.assertRaisesRegex(ValueError, "temporary directory"):
            self._run(remote_name)
        self.assertEqual(self.downloads, [])
        self.assertEqual(self._files_outside_tmp_dirs(), [])

    def test_absolute_name_is_rejected(self):
        target = self.root / "elsewhere" / "outside.zip"
        with self.assertRaisesRegex(ValueError, "temporary directory"):
            self._run(str(target))
        self.assertFalse(target.exists())
        self.assertEqual(self.downloads, [])

    def test_plain_name_round_trip(self):
        self._run("archive.zip")
        self.assertEqual(self.downloads[0].name, "archive.zip")
        self.assertEqual(self.downloads[0].parent.parent, self.tmp_base)
        self.assertEqual((self.output_dir / "archive" / "data.txt").read_text(), "payload")

    def test_remote_name_requested_once(self):
        """The extension check and the download destination use the same server response."""
        _, head = self._run("archive.zip")
        self.assertEqual(head.call_count, 1)

    @parameterized.expand([("dotdot", ".."), ("dot", "."), ("slash", "/"), ("doubleslash", "//")])
    def test_name_without_basename_is_rejected(self, _, remote_name):
        with self.assertRaisesRegex(ValueError, "temporary directory"):
            self._run(remote_name)
        self.assertEqual(self.downloads, [])
        self.assertFalse(self.output_dir.exists())

    def test_explicit_filepath_is_unchanged(self):
        filepath = self.root / "cache" / "archive.zip"
        head = mock.Mock(return_value=_head_response("../ignored.zip"))
        with (
            mock.patch.object(utils.requests, "head", head),
            mock.patch.object(utils, "download_url", side_effect=self._fake_download),
        ):
            utils.download_and_extract(
                "https://example.invalid/archive.zip", filepath=filepath, output_dir=self.output_dir
            )
        self.assertEqual(self.downloads, [filepath])
        self.assertTrue(filepath.is_file())


if __name__ == "__main__":
    unittest.main()
