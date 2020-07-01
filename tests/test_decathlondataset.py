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

from monai.apps import DecathlonDataset
from monai.transforms import LoadNiftid, AddChanneld, ScaleIntensityd, ToTensord, Compose
from tests.utils import skip_if_quick


class TestDecathlonDataset(unittest.TestCase):
    @skip_if_quick
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

        try:
            data = DecathlonDataset(
                root_dir=tempdir, task="Task04_Hippocampus", transform=transform, section="validation", download=True
            )
        except RuntimeError as e:
            if str(e).startswith("download failed due to network issue or permission denied."):
                shutil.rmtree(tempdir)
                return

        _test_dataset(data)
        data = DecathlonDataset(
            root_dir=tempdir, task="Task04_Hippocampus", transform=transform, section="validation", download=False
        )
        _test_dataset(data)
        shutil.rmtree(os.path.join(tempdir, "Task04_Hippocampus"))
        try:
            data = DecathlonDataset(
                root_dir=tempdir, task="Task04_Hippocampus", transform=transform, section="validation", download=False
            )
        except RuntimeError as e:
            print(str(e))
            self.assertTrue(str(e).startswith("can not find dataset directory"))

        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
