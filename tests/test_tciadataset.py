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
import shutil
import unittest

from monai.apps import TciaDataset
from monai.apps.tcia import TCIA_LABEL_DICT
from monai.data import MetaTensor
from monai.transforms import AddChanneld, Compose, LoadImaged, ScaleIntensityd
from tests.utils import skip_if_downloading_fails, skip_if_quick


class TestTciaDataset(unittest.TestCase):
    @skip_if_quick
    def test_values(self):
        testing_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        download_len = 1
        val_frac = 1.0
        collection = "QIN-PROSTATE-Repeatability"

        transform = Compose(
            [
                LoadImaged(keys=["image", "seg"], reader="PydicomReader", label_dict=TCIA_LABEL_DICT[collection]),
                AddChanneld(keys="image"),
                ScaleIntensityd(keys="image"),
            ]
        )

        def _test_dataset(dataset):
            self.assertEqual(len(dataset), int(download_len * val_frac))
            self.assertTrue("image" in dataset[0])
            self.assertTrue("seg" in dataset[0])
            self.assertTrue(isinstance(dataset[0]["image"], MetaTensor))
            self.assertTupleEqual(dataset[0]["image"].shape, (1, 256, 256, 24))
            self.assertTupleEqual(dataset[0]["seg"].shape, (256, 256, 24, 4))

        with skip_if_downloading_fails():
            data = TciaDataset(
                root_dir=testing_dir,
                collection=collection,
                transform=transform,
                section="validation",
                download=True,
                download_len=download_len,
                copy_cache=False,
                val_frac=val_frac,
            )

        _test_dataset(data)
        data = TciaDataset(
            root_dir=testing_dir,
            collection=collection,
            transform=transform,
            section="validation",
            download=False,
            val_frac=val_frac,
        )
        _test_dataset(data)
        self.assertTrue(
            data[0]["image"].meta["filename_or_obj"].endswith("QIN-PROSTATE-Repeatability/PCAMPMRI-00015/1901/image")
        )
        self.assertTrue(
            data[0]["seg"].meta["filename_or_obj"].endswith("QIN-PROSTATE-Repeatability/PCAMPMRI-00015/1901/seg")
        )
        # test validation without transforms
        data = TciaDataset(
            root_dir=testing_dir, collection=collection, section="validation", download=False, val_frac=val_frac
        )
        self.assertTupleEqual(data[0]["image"].shape, (256, 256, 24))
        self.assertEqual(len(data), int(download_len * val_frac))
        data = TciaDataset(
            root_dir=testing_dir, collection=collection, section="validation", download=False, val_frac=val_frac
        )
        self.assertTupleEqual(data[0]["image"].shape, (256, 256, 24))
        self.assertEqual(len(data), download_len)

        shutil.rmtree(os.path.join(testing_dir, collection))
        try:
            TciaDataset(
                root_dir=testing_dir,
                collection=collection,
                transform=transform,
                section="validation",
                download=False,
                val_frac=val_frac,
            )
        except RuntimeError as e:
            self.assertTrue(str(e).startswith("Cannot find dataset directory"))


if __name__ == "__main__":
    unittest.main()
