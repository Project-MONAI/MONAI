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

from monai.data import ITKWriter


class TestITKWriter(unittest.TestCase):
    def test_saved_content(self):
        with tempfile.TemporaryDirectory() as tempdir:

            writer = ITKWriter(output_dir=tempdir, output_postfix="seg", output_ext=".dcm", output_dtype=np.uint8)

            meta_data = {"filename_or_obj": ["testfile" + str(i) + ".nii" for i in range(8)]}
            writer.write_batch(torch.zeros(8, 1, 2, 3, 4), meta_data)
            for i in range(8):
                filepath = os.path.join("testfile" + str(i), "testfile" + str(i) + "_seg.dcm")
                self.assertTrue(os.path.exists(os.path.join(tempdir, filepath)))


if __name__ == "__main__":
    unittest.main()
