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
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from parameterized import parameterized

from monai.tests.utils import TEST_NDARRAYS, make_nifti_image
from monai.transforms import Compose, LoadImage, MappingJson, SaveImage

TESTS = []
for p in TEST_NDARRAYS:
    for q in TEST_NDARRAYS:
        TEST_IMAGE = p(np.arange(24).reshape((2, 4, 3)))
        TEST_AFFINE = q(
            np.array(
                [[-5.3, 0.0, 0.0, 102.01], [0.0, 0.52, 2.17, -7.50], [-0.0, 1.98, -0.26, -23.12], [0.0, 0.0, 0.0, 1.0]]
            )
        )
        TESTS.append([TEST_IMAGE, TEST_AFFINE, True])
        TESTS.append([TEST_IMAGE, TEST_AFFINE, False])


class TestMappingJson(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mapping_json_path = os.path.join(self.test_dir, "mapping.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @parameterized.expand(TESTS)
    def test_mapping_json(self, array, affine, savepath_in_metadict):
        name = "test_image"
        output_ext = ".nii.gz"
        test_image_name = make_nifti_image(array, affine, fname=os.path.join(self.test_dir, name))

        input_file = os.path.join(self.test_dir, test_image_name)
        output_file = os.path.join(self.test_dir, name, name + "_trans" + output_ext)

        transforms = Compose(
            [
                LoadImage(reader="NibabelReader", image_only=True),
                SaveImage(output_dir=self.test_dir, output_ext=output_ext, savepath_in_metadict=savepath_in_metadict),
                MappingJson(mapping_json_path=self.mapping_json_path),
            ]
        )

        if savepath_in_metadict:
            transforms(input_file)
            self.assertTrue(Path(self.mapping_json_path).exists())
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
