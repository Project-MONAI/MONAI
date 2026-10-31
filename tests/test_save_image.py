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

import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pytest

from monai.data import image_writer
from monai.data.image_writer import ImageWriter, resolve_writer
from monai.utils import OptionalImportError


def _reset_writers():
    """Restore ``SUPPORTED_WRITERS`` to the state created by ``image_writer.init()``."""
    image_writer.SUPPORTED_WRITERS.clear()
    image_writer.init()


class TestResolveWriterMissingBackend(unittest.TestCase):
    """The error message should tell which package to install."""

    def setUp(self):
        _reset_writers()

    def tearDown(self):
        _reset_writers()

    def test_missing_pillow(self):
        # ``PILWriter`` is the only backend for png, the pip package name is ``pillow`` rather than ``PIL``
        with patch.dict(sys.modules, {"PIL": None}):
            with self.assertRaises(OptionalImportError) as ctx:
                resolve_writer(".png")
        msg = str(ctx.exception)
        self.assertIn("No ImageWriter backend found for png", msg)
        self.assertIn("`pip install pillow`", msg)
        self.assertNotIn("`pip install PIL`", msg)

    def test_missing_itk_and_nibabel(self):
        with patch.dict(sys.modules, {"itk": None, "nibabel": None}):
            with self.assertRaises(OptionalImportError) as ctx:
                resolve_writer(".nii.gz")
        msg = str(ctx.exception)
        self.assertIn("No ImageWriter backend found for nii.gz", msg)
        self.assertIn("`pip install itk`", msg)
        self.assertIn("`pip install nibabel`", msg)

    def test_missing_backend_is_import_error(self):
        # the exception type is unchanged, only its message is extended
        with patch.dict(sys.modules, {"itk": None, "nibabel": None}):
            with self.assertRaises(ImportError):
                resolve_writer(".nii")

    def test_custom_writer_without_package_hint(self):
        # a custom backend that provides no package hint keeps the previous message
        class _CustomWriter(ImageWriter):
            def __init__(self, *args, **kwargs):
                raise OptionalImportError("required package `custom-pkg` is not installed.")

        try:
            image_writer.register_writer("custom", _CustomWriter)
            with self.assertRaises(OptionalImportError) as ctx:
                resolve_writer(".custom")
            self.assertEqual(str(ctx.exception), "No ImageWriter backend found for custom.")
        finally:
            image_writer.SUPPORTED_WRITERS.pop("custom", None)

    def test_required_package_of_built_in_writers(self):
        self.assertEqual(image_writer.ITKWriter.required_package, "itk")
        self.assertEqual(image_writer.NibabelWriter.required_package, "nibabel")
        self.assertEqual(image_writer.PILWriter.required_package, "pillow")
        self.assertIsNone(ImageWriter.required_package)


def test_save_image_missing_backend():
    """``SaveImage`` reports the install command when no writer backend is available."""
    from monai.transforms import SaveImage

    out_dir = tempfile.mkdtemp()
    _reset_writers()
    try:
        with patch.dict(sys.modules, {"itk": None, "nibabel": None}):
            with pytest.raises(ImportError, match="pip install itk"):
                SaveImage(output_dir=out_dir, output_ext=".nii.gz")
    finally:
        _reset_writers()
        shutil.rmtree(out_dir, ignore_errors=True)
