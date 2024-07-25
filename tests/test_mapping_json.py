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

import numpy as np
from parameterized import parameterized

from monai.transforms import Compose, LoadImage, MappingJson, SaveImage
from monai.utils import optional_import

nib, has_nib = optional_import("nibabel")


@unittest.skipUnless(has_nib, "nibabel required")
class TestMappingJsonD(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mapping_json_path = os.path.join(self.temp_dir.name, "mapping.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    @parameterized.expand([(True,), (False,)])
    def test_mapping_jsond(self, savepath_in_metadict):
        test_image = np.random.rand(128, 128, 128)
        output_ext = ".nii.gz"
        name = "test_image"

        input_file = os.path.join(self.temp_dir.name, name + output_ext)
        output_file = os.path.join(self.temp_dir.name, name, name + "_trans" + output_ext)

        nib.save(nib.Nifti1Image(test_image, np.eye(4)), input_file)

        transforms = Compose(
            [
                LoadImage(image_only=True),
                SaveImage(
                    output_dir=self.temp_dir.name, output_ext=output_ext, savepath_in_metadict=savepath_in_metadict
                ),
                MappingJson(mapping_json_path=self.mapping_json_path),
            ]
        )

        if savepath_in_metadict:
            transforms(input_file)
            self.assertTrue(os.path.exists(self.mapping_json_path))
            with open(self.mapping_json_path) as f:
                mapping_data = json.load(f)

            self.assertEqual(len(mapping_data), 1)
            self.assertEqual(mapping_data[0]["input"], input_file)
            self.assertEqual(mapping_data[0]["output"], output_file)
        else:
            with self.assertRaises(RuntimeError) as cm:
                transforms(input_file)
            the_exception = cm.exception
            cause_exception = the_exception.__cause__

            self.assertIsInstance(cause_exception, KeyError)
            self.assertIn(
                "Missing 'saved_to' key in metadata. Check SaveImage savepath_in_metadict.", str(cause_exception)
            )


if __name__ == "__main__":
    unittest.main()
