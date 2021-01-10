# Copyright 2020 - 2021 MONAI Consortium
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

import numpy as np
import torch

from monai.data import NiftiSaver


class TestNiftiSaver(unittest.TestCase):
    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:

            saver = NiftiSaver(output_dir=tempdir, output_postfix="seg", output_ext=".nii.gz")

            meta_data = {"filename_or_obj": ["testfile" + str(i) + ".nii" for i in range(8)]}
            saver.save_batch(torch.zeros(8, 1, 2, 2), meta_data)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.nii.gz")
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))

    def test_saved_resize_content(self):
        with tempfile.TemporaryDirectory() as tempdir:

            saver = NiftiSaver(
                output_dir=tempdir,
                output_postfix="seg",
                output_ext=".nii.gz",
                dtype=np.float32,
            )

            meta_data = {
                "filename_or_obj": ["testfile" + str(i) + ".nii" for i in range(8)],
                "affine": [np.diag(np.ones(4)) * 5] * 8,
                "original_affine": [np.diag(np.ones(4)) * 1.0] * 8,
            }
            saver.save_batch(torch.randint(0, 255, (8, 8, 2, 2)), meta_data)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.nii.gz")
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))

    def test_saved_3d_resize_content(self):
        with tempfile.TemporaryDirectory() as tempdir:

            saver = NiftiSaver(
                output_dir=tempdir,
                output_postfix="seg",
                output_ext=".nii.gz",
                dtype=np.float32,
            )

            meta_data = {
                "filename_or_obj": ["testfile" + str(i) + ".nii.gz" for i in range(8)],
                "spatial_shape": [(10, 10, 2)] * 8,
                "affine": [np.diag(np.ones(4)) * 5] * 8,
                "original_affine": [np.diag(np.ones(4)) * 1.0] * 8,
            }
            saver.save_batch(torch.randint(0, 255, (8, 8, 1, 2, 2)), meta_data)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.nii.gz")
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))


if __name__ == "__main__":
    unittest.main()
