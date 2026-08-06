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
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from monai.apps import create_temp_dir
from monai.utils import MONAIEnvVars

MONAI_DATA_DIRECTORY = "MONAI_DATA_DIRECTORY"


class TestCreateTempDir(unittest.TestCase):
    def test_basic_use(self):
        """Test basic usage which should create a new random temporary directory."""

        data_dir = os.environ.pop(MONAI_DATA_DIRECTORY, None)  # ignore the environment variable if present
        try:
            with patch("atexit.register") as mock_reg:
                test_dir = create_temp_dir()

                self.assertTrue(os.path.isdir(test_dir))

                mock_reg.assert_called_once_with(shutil.rmtree, test_dir, ignore_errors=True)
        finally:
            if data_dir is not None:
                os.environ[MONAI_DATA_DIRECTORY] = data_dir

    def test_data_dir(self):
        """Test using a mocked MONAI_DATA_DIRECTORY, which should be returned by the function."""
        with patch("monai.utils.MONAIEnvVars.data_dir") as data_dir, tempfile.TemporaryDirectory() as fake_data_dir:
            data_dir.return_value = fake_data_dir

            self.assertEqual(fake_data_dir, MONAIEnvVars.data_dir())

            test_dir = create_temp_dir()

            self.assertTrue(os.path.isdir(test_dir))
            self.assertEqual(test_dir, fake_data_dir)

    def test_given_dir(self):
        """Test giving a directory to the function, ensuring it creates the directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_dir = f"{temp_dir}{os.path.sep}test_inner_dir"

            test_dir = create_temp_dir(selected_dir)

            self.assertTrue(os.path.isdir(selected_dir))
            self.assertEqual(test_dir, selected_dir)
            self.assertEqual(test_dir, selected_dir)

    def test_given_dir_path(self):
        """Test giving a directory as a Path object to the function, ensuring it creates the directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_dir = f"{temp_dir}{os.path.sep}test_inner_dir"

            test_dir = create_temp_dir(Path(selected_dir))

            self.assertTrue(os.path.isdir(selected_dir))
            self.assertEqual(test_dir, selected_dir)

    def test_finalisation(self):
        """Test the temporary directory is deleted by finalisation."""
        self.finaliser = None

        def _register(func, /, *args, **kwargs):
            self.finaliser = (func, args, kwargs)

        with patch("atexit.register", new=_register), tempfile.TemporaryDirectory() as temp_dir:
            selected_dir = f"{temp_dir}{os.path.sep}test_inner_dir"
            test_dir = create_temp_dir(selected_dir, True)

            self.assertTrue(os.path.isdir(selected_dir))
            self.assertIsNotNone(self.finaliser)

            with open(test_dir + "/test_file", "w") as o:
                o.write("Test file data")

            func, args, kwargs = self.finaliser
            func(*args, **kwargs)

            self.assertFalse(os.path.exists(selected_dir))


if __name__ == "__main__":
    unittest.main()
