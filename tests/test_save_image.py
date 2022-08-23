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

import torch
from parameterized import parameterized

from monai.data.meta_tensor import MetaTensor
from monai.transforms import SaveImage

TEST_CASE_1 = [torch.randint(0, 255, (1, 2, 3, 4)), {"filename_or_obj": "testfile0.nii.gz"}, ".nii.gz", False]

TEST_CASE_2 = [torch.randint(0, 255, (1, 2, 3, 4)), None, ".nii.gz", False]

TEST_CASE_3 = [torch.randint(0, 255, (1, 2, 3, 4)), {"filename_or_obj": "testfile0.nrrd"}, ".nrrd", False]

TEST_CASE_4 = [
    torch.randint(0, 255, (3, 2, 4, 5), dtype=torch.uint8),
    {"filename_or_obj": "testfile0.dcm"},
    ".dcm",
    False,
]


class TestSaveImage(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3, TEST_CASE_4])
    def test_saved_content(self, test_data, meta_data, output_ext, resample):
        if meta_data is not None:
            test_data = MetaTensor(test_data, meta=meta_data)

        with tempfile.TemporaryDirectory() as tempdir:
            trans = SaveImage(
                output_dir=tempdir,
                output_ext=output_ext,
                resample=resample,
                separate_folder=False,  # test saving into the same folder
            )
            trans(test_data)

            filepath = "testfile0" if meta_data is not None else "0"
            self.assertTrue(os.path.exists(os.path.join(tempdir, filepath + "_trans" + output_ext)))


if __name__ == "__main__":
    unittest.main()
