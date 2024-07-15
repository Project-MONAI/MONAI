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

import json
import os
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import NibabelReader
from monai.transforms import Compose, LoadImage, SaveImage
from monai.transforms.io.array import MappingJson


class TestMappingJson(unittest.TestCase):
    def setUp(self):
        self.mapping_json_path = "test_mapping.json"
        if Path(self.mapping_json_path).exists():
            Path(self.mapping_json_path).unlink()

    TEST_CASE_1 = [{}, ["test_image.nii.gz"], (128, 128, 128), True]
    TEST_CASE_2 = [{}, ["test_image.nii.gz"], (128, 128, 128), False]

    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_mapping_json(self, load_params, filenames, expected_shape, savepath_in_metadict):
        test_image = np.random.rand(128, 128, 128)

        with tempfile.TemporaryDirectory() as tempdir:
            for i, name in enumerate(filenames):
                file_path = os.path.join(tempdir, name)
                nib.save(nib.Nifti1Image(test_image, np.eye(4)), file_path)
                filenames[i] = file_path

            transforms = Compose(
                [
                    LoadImage(image_only=True, **load_params),
                    SaveImage(output_dir=tempdir, output_ext=".nii.gz", savepath_in_metadict=savepath_in_metadict),
                    MappingJson(mapping_json_path=self.mapping_json_path),
                ]
            )

            if savepath_in_metadict:
                result = transforms(filenames[0])

                img = result
                meta = img.meta

                self.assertEqual(img.shape, expected_shape)

                self.assertTrue(Path(self.mapping_json_path).exists())
                with open(self.mapping_json_path) as f:
                    mapping_data = json.load(f)

                expected_mapping = [{"input": meta["filename_or_obj"], "output": meta["saved_to"]}]
                self.assertEqual(expected_mapping, mapping_data)
            else:
                with self.assertRaises(RuntimeError) as cm:
                    transforms(filenames[0])
                the_exception = cm.exception
                self.assertIsInstance(the_exception.__cause__, KeyError)
                self.assertIn(
                    "Missing 'saved_to' key in metadata. Check SaveImage savepath_in_metadict.",
                    str(the_exception.__cause__),
                )


if __name__ == "__main__":
    unittest.main()
