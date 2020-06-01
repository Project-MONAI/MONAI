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

import unittest
import os
import shutil
import numpy as np
import tempfile
from PIL import Image
from monai.data import load_decathalon_datalist


class TestLoadDecathalonDatalist(unittest.TestCase):
    def test_values(self):
        test_data = {
            "name": "Spleen",
            "description": "Spleen Segmentation",
            "labels": {"0": "background", "1": "spleen"},
            "training": [
                {
                    "image": "./imagesTr/spleen_19.nii.gz",
                    "label": "./labelsTr/spleen_19.nii.gz"
                },
                {
                    "image": "./imagesTr/spleen_31.nii.gz",
                    "label": "./labelsTr/spleen_31.nii.gz"
                }
            ],
            "test": [
                "./imagesTs/spleen_15.nii.gz",
                "./imagesTs/spleen_23.nii.gz"
            ]
        }
        test_image = np.random.randint(0, 256, size=data_shape)
        tempdir = tempfile.mkdtemp()
        for i, name in enumerate(filenames):
            filenames[i] = os.path.join(tempdir, name)
            Image.fromarray(test_image.astype("uint8")).save(filenames[i])
        result = LoadPNG()(filenames)
        self.assertTupleEqual(result[1]["spatial_shape"], meta_shape)
        self.assertTupleEqual(result[0].shape, expected_shape)
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
