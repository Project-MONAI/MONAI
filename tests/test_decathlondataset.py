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
import tempfile

from monai.application import DecathlonDataset
from tests.utils import NumpyImageTestCase2D
from monai.transforms import LoadNiftid, AddChanneld, ScaleIntensityd, ToTensord, Compose
# from tests.utils import skip_if_quick

from monai.utils import optional_import
from monai.application import check_md5
gdown, has_gdown = optional_import("gdown", "3.11.1")


class TestDecathlonDataset(unittest.TestCase):
    # @skip_if_quick
    def test_values(self):
        tempdir = tempfile.mkdtemp()
        transform = Compose(
            [
                LoadNiftid(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        def _test_dataset(dataset):
            self.assertEqual(len(dataset), 52)
            self.assertTrue("image" in dataset[0])
            self.assertTrue("label" in dataset[0])
            self.assertTrue("image_meta_dict" in dataset[0])
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 33, 47, 34))

        data = DecathlonDataset(
            root_dir=tempdir,
            task="Task04_Hippocampus",
            transform=transform,
            section="validation",
            download=True
        )
        _test_dataset(data)

        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
