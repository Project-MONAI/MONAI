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

import os
import tempfile
import unittest
from pathlib import Path

import torch

from monai.data import PNGSaver


class TestPNGSaver(unittest.TestCase):
    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:

            saver = PNGSaver(output_dir=tempdir, output_postfix="seg", output_ext=".png", scale=255)

            meta_data = {"filename_or_obj": ["testfile" + str(i) + ".jpg" for i in range(8)]}
            saver.save_batch(torch.randint(1, 200, (8, 1, 2, 2)), meta_data)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.png")
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))

    def test_saved_content_three_channel(self):
        with tempfile.TemporaryDirectory() as tempdir:

            saver = PNGSaver(output_dir=Path(tempdir), output_postfix="seg", output_ext=".png", scale=255)

            meta_data = {"filename_or_obj": ["testfile" + str(i) + ".jpg" for i in range(8)]}
            saver.save_batch(torch.randint(1, 200, (8, 3, 2, 2)), meta_data)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.png")
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))

    def test_saved_content_spatial_size(self):
        with tempfile.TemporaryDirectory() as tempdir:

            saver = PNGSaver(output_dir=tempdir, output_postfix="seg", output_ext=".png", scale=255)

            meta_data = {
                "filename_or_obj": ["testfile" + str(i) + ".jpg" for i in range(8)],
                "spatial_shape": [(4, 4) for i in range(8)],
            }
            saver.save_batch(torch.randint(1, 200, (8, 1, 2, 2)), meta_data)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.png")
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))

    def test_saved_specified_root(self):
        with tempfile.TemporaryDirectory() as tempdir:

            saver = PNGSaver(
                output_dir=tempdir, output_postfix="seg", output_ext=".png", scale=255, data_root_dir="test"
            )

            meta_data = {
                "filename_or_obj": [os.path.join("test", "testfile" + str(i), "image" + ".jpg") for i in range(8)]
            }
            saver.save_batch(torch.randint(1, 200, (8, 1, 2, 2)), meta_data)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "image", "image" + "_seg.png")
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))


if __name__ == "__main__":
    unittest.main()
