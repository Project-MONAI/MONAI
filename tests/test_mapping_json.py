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
import multiprocessing
import os
import random
import shutil
import tempfile
import time
import unittest

import numpy as np
from parameterized import parameterized

from monai.transforms import Compose, LoadImage, MappingJson, SaveImage
from monai.utils import optional_import

nib, has_nib = optional_import("nibabel")


def create_input_file(temp_dir, name):
    test_image = np.random.rand(128, 128, 128)
    output_ext = ".nii.gz"
    input_file = os.path.join(temp_dir, name + output_ext)
    nib.save(nib.Nifti1Image(test_image, np.eye(4)), input_file)
    return input_file


def create_transform(temp_dir, mapping_json_path, savepath_in_metadict=True):
    return Compose(
        [
            LoadImage(image_only=True),
            SaveImage(output_dir=temp_dir, output_ext=".nii.gz", savepath_in_metadict=savepath_in_metadict),
            MappingJson(mapping_json_path=mapping_json_path),
        ]
    )


def process_image(args):
    temp_dir, mapping_json_path, i = args
    time.sleep(random.uniform(0, 0.1))
    input_file = create_input_file(temp_dir, f"test_image_{i}")
    transform = create_transform(temp_dir, mapping_json_path)
    transform(input_file)
    time.sleep(random.uniform(0, 0.1))


@unittest.skipUnless(has_nib, "nibabel required")
class TestMappingJson(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mapping_json_path = os.path.join(self.temp_dir, "mapping.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @parameterized.expand([(True,), (False,)])
    def test_mapping_json(self, savepath_in_metadict):
        name = "test_image"
        input_file = create_input_file(self.temp_dir, name)
        output_file = os.path.join(self.temp_dir, name, name + "_trans.nii.gz")

        transform = create_transform(self.temp_dir, self.mapping_json_path, savepath_in_metadict)

        if savepath_in_metadict:
            transform(input_file)
            self.assertTrue(os.path.exists(self.mapping_json_path))
            with open(self.mapping_json_path) as f:
                mapping_data = json.load(f)
            self.assertEqual(len(mapping_data), 1)
            self.assertEqual(mapping_data[0]["input"], input_file)
            self.assertEqual(mapping_data[0]["output"], output_file)
        else:
            with self.assertRaises(RuntimeError) as cm:
                transform(input_file)
            cause_exception = cm.exception.__cause__
            self.assertIsInstance(cause_exception, KeyError)
            self.assertIn(
                "Missing 'saved_to' key in metadata. Check SaveImage savepath_in_metadict.", str(cause_exception)
            )

    def test_multiprocess_mapping_json(self):
        num_processes, num_images = 16, 1000

        with multiprocessing.Pool(processes=num_processes) as pool:
            args = [(self.temp_dir, self.mapping_json_path, i) for i in range(num_images)]
            pool.map(process_image, args)

        with open(self.mapping_json_path) as f:
            mapping_data = json.load(f)

        self.assertEqual(len(mapping_data), num_images, f"Expected {num_images} entries, but got {len(mapping_data)}")
        unique_entries = set(tuple(sorted(entry.items())) for entry in mapping_data)
        self.assertEqual(len(mapping_data), len(unique_entries), "Duplicate entries exist")
        for entry in mapping_data:
            self.assertIn("input", entry, "Entry missing 'input' key")
            self.assertIn("output", entry, "Entry missing 'output' key")


if __name__ == "__main__":
    unittest.main()
