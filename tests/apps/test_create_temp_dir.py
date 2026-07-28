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
import sys
import tempfile
import unittest
from subprocess import call
from textwrap import dedent
from unittest.mock import patch

import monai
from monai.apps import create_temp_dir
from monai.utils import MONAIEnvVars
from tests.test_utils import skip_if_quick

MONAI_DATA_DIRECTORY = "MONAI_DATA_DIRECTORY"


class TestCreateTempDir(unittest.TestCase):
    def test_basic_use(self):
        """Test basic usage which should create a new random temporary directory."""
        try:
            data_dir = os.environ.pop(MONAI_DATA_DIRECTORY, None)  # ignore the environment variable if present

            test_dir = create_temp_dir()

            self.assertTrue(os.path.isdir(test_dir))

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
            selected_dir = f"{temp_dir}/test_inner_dir"
            test_dir = create_temp_dir(selected_dir)

            self.assertTrue(os.path.isdir(selected_dir))
            self.assertEqual(test_dir, selected_dir)

    @skip_if_quick
    def test_finalisation(self):
        """Test the temporary directory is deleted by finalisation using a subprocess."""
        # this writes a script to a temporary directory which uses create_temp_dir, this is then called in a subprocess
        # to verify the finalising behaviour
        with tempfile.TemporaryDirectory() as temp_dir:
            selected_dir = f"{temp_dir}/test_inner_dir"
            script = f"""
                import os
                from monai.apps import create_temp_dir
                test_dir = create_temp_dir({selected_dir!r}, True)
                with open(test_dir+"/test_file", "w") as o:
                    o.write("Test file data")
                assert os.path.isdir(test_dir)
            """
            script_file = f"{temp_dir}/script.py"
            with open(script_file, "w") as o:
                o.write(dedent(script))

            # add a PYTHONPATH value to the environment to find the current MONAI install
            env = {**os.environ, "PYTHONPATH": f"{os.path.dirname(monai.__file__)}/.."}
            retcode = call([sys.executable, "script.py"], cwd=temp_dir, env=env)

            self.assertEqual(0, retcode)
            self.assertFalse(os.path.exists(selected_dir))


if __name__ == "__main__":
    unittest.main()
