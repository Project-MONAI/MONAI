# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import unittest
import tempfile

import datetime

from monai.utils.misc import create_run_dir


class TestCreateRunDir(unittest.TestCase):
    def setUp(self):
        self.root_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root_dir)

    def test_create_run_dir(self):
        dir1 = create_run_dir(self.root_dir)
        self.assertTrue(os.path.isdir(dir1))

    def test_override_time(self):
        test_time = datetime.datetime(1995, 3, 31, 5, 35, 0)
        true_output = os.path.join(self.root_dir, "model_1995_03_31_05_35_00")
        dir1 = create_run_dir(self.root_dir, time=test_time)
        self.assertEqual(dir1, true_output, "Created unexpected rundir name.")

    def test_change_prefix(self):
        test_time = datetime.datetime(2000, 1, 1)
        test_prefix = "MONAI"
        dir1 = create_run_dir(self.root_dir, prefix=test_prefix, time=test_time)
        test_dirname = os.path.join(self.root_dir, "MONAI_2000_01_01_00_00_00")
        self.assertEqual(dir1, test_dirname, "Did not change rundir prefix.")

    def test_change_timeformat(self):
        test_time = datetime.datetime(1234, 5, 6)
        test_format = "%d_%m_%Y"
        test_dirname = os.path.join(self.root_dir, "model_06_05_1234")
        dir1 = create_run_dir(self.root_dir, time_format=test_format, time=test_time)
        self.assertEqual(dir1, test_dirname)

    def test_create_multiple_run_dir(self):
        time1 = datetime.datetime.now()
        dir1 = create_run_dir(self.root_dir, time=time1)
        time2 = datetime.datetime(1337, 1, 1)
        dir2 = create_run_dir(self.root_dir, time=time2)
        self.assertTrue(os.path.isdir(dir1))
        self.assertTrue(os.path.isdir(dir2))
        self.assertNotEqual(dir1, dir2, "Created same directory")

    def test_make_save_dir(self):
        test_time = datetime.datetime(1995, 3, 31, 5, 35, 0)
        new_save_dir = os.path.join(self.root_dir, "monai_model_output")
        true_output = os.path.join(new_save_dir, "model_1995_03_31_05_35_00")
        dir1 = create_run_dir(new_save_dir, time=test_time)
        self.assertTrue(os.path.isdir(dir1), "Did not create rundir folder in custom save_dir.")
        self.assertEqual(dir1, true_output, "Did not name save_dir properly.")


if __name__ == "__main__":
    unittest.main()
