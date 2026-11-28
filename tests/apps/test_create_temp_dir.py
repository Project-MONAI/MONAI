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

from parameterized import parameterized

from monai.apps import create_temp_dir
from monai.utils import MONAIEnvVars

MONAI_DATA_DIRECTORY = "MONAI_DATA_DIRECTORY"


class TestCreateTempDir(unittest.TestCase):
    def test_basic_use(self):
        """Test basic usage which should create a new random temporary directory."""

        data_dir = os.environ.pop(MONAI_DATA_DIRECTORY, None)  # don't let environment variable interfere with testing
        test_dir = None
        try:
            with patch("atexit.register") as mock_reg:
                test_dir = create_temp_dir()

                self.assertTrue(os.path.isdir(test_dir))

                mock_reg.assert_called_once_with(shutil.rmtree, test_dir, ignore_errors=True)
        finally:
            if data_dir is not None:
                os.environ[MONAI_DATA_DIRECTORY] = data_dir
            if test_dir is not None:
                shutil.rmtree(test_dir, ignore_errors=True)

    def test_data_dir(self):
        """Test using a mocked MONAIEnvVars.data_dir(), which should be returned by the function."""
        with (
            patch("atexit.register") as mock_reg,
            patch("monai.utils.MONAIEnvVars.data_dir") as data_dir,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            data_dir.return_value = selected_dir = f"{temp_dir}{os.path.sep}test_inner_dir"

            self.assertEqual(selected_dir, MONAIEnvVars.data_dir())

            test_dir = create_temp_dir()

            mock_reg.assert_not_called()
            self.assertTrue(os.path.isdir(test_dir))
            self.assertEqual(test_dir, selected_dir)

    @parameterized.expand([(False, False), (True, False), (False, True)])
    def test_given_dir(self, as_path, delete_on_finalise):
        """
        Test giving a directory to the function as string or Path, ensuring it creates the directory. If
        `delete_on_finalise` is True, check the call to rmtree was registered, otherwise check atexit was not called.
        """
        with patch("atexit.register") as mock_reg, tempfile.TemporaryDirectory() as temp_dir:
            selected_dir = f"{temp_dir}{os.path.sep}test_inner_dir"

            test_dir = create_temp_dir(Path(selected_dir) if as_path else selected_dir, delete_on_finalise)

            self.assertTrue(os.path.isdir(selected_dir))
            self.assertEqual(test_dir, selected_dir)

            if delete_on_finalise:
                mock_reg.assert_called_once_with(shutil.rmtree, selected_dir, ignore_errors=True)
            else:
                mock_reg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
