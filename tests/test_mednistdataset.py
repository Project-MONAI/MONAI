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
import shutil
import unittest
from urllib.error import ContentTooShortError, HTTPError

from monai.apps import MedNISTDataset
from monai.transforms import AddChanneld, Compose, LoadImaged, ScaleIntensityd, ToTensord
from tests.utils import skip_if_quick


class TestMedNISTDataset(unittest.TestCase):
    @skip_if_quick
    def test_values(self):
        testing_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        transform = Compose(
            [
                LoadImaged(keys="image"),
                AddChanneld(keys="image"),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        def _test_dataset(dataset):
            self.assertEqual(len(dataset), 5986)
            self.assertTrue("image" in dataset[0])
            self.assertTrue("label" in dataset[0])
            self.assertTrue("image_meta_dict" in dataset[0])
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 64, 64))

        try:  # will start downloading if testing_dir doesn't have the MedNIST files
            data = MedNISTDataset(root_dir=testing_dir, transform=transform, section="test", download=True)
        except (ContentTooShortError, HTTPError, RuntimeError) as e:
            print(str(e))
            if isinstance(e, RuntimeError):
                # FIXME: skip MD5 check as current downloading method may fail
                self.assertTrue(str(e).startswith("md5 check"))
            return  # skipping this test due the network connection errors

        _test_dataset(data)

        # testing from
        data = MedNISTDataset(root_dir=testing_dir, transform=transform, section="test", download=False)
        _test_dataset(data)
        data = MedNISTDataset(root_dir=testing_dir, section="test", download=False)
        self.assertTupleEqual(data[0]["image"].shape, (64, 64))
        shutil.rmtree(os.path.join(testing_dir, "MedNIST"))
        try:
            data = MedNISTDataset(root_dir=testing_dir, transform=transform, section="test", download=False)
        except RuntimeError as e:
            print(str(e))
            self.assertTrue(str(e).startswith("Cannot find dataset directory"))


if __name__ == "__main__":
    unittest.main()
