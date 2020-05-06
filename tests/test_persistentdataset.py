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
import nibabel as nib
from parameterized import parameterized
from monai.data import PersistentDataset
from monai.transforms import Compose, LoadNiftid, SimulateDelayd

TEST_CASE_1 = [(128, 128, 128)]


class TestDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1])
    def test_shape(self, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]), np.eye(4))
        tempdir = tempfile.mkdtemp()
        nib.save(test_image, os.path.join(tempdir, "test_image1.nii.gz"))
        nib.save(test_image, os.path.join(tempdir, "test_label1.nii.gz"))
        nib.save(test_image, os.path.join(tempdir, "test_extra1.nii.gz"))
        nib.save(test_image, os.path.join(tempdir, "test_image2.nii.gz"))
        nib.save(test_image, os.path.join(tempdir, "test_label2.nii.gz"))
        nib.save(test_image, os.path.join(tempdir, "test_extra2.nii.gz"))
        test_data = [
            {
                "image": os.path.join(tempdir, "test_image1.nii.gz"),
                "label": os.path.join(tempdir, "test_label1.nii.gz"),
                "extra": os.path.join(tempdir, "test_extra1.nii.gz"),
            },
            {
                "image": os.path.join(tempdir, "test_image2.nii.gz"),
                "label": os.path.join(tempdir, "test_label2.nii.gz"),
                "extra": os.path.join(tempdir, "test_extra2.nii.gz"),
            },
        ]
        test_transform = Compose(
            [
                LoadNiftid(keys=["image", "label", "extra"]),
                SimulateDelayd(keys=["image", "label", "extra"], delay_time=[1e-7, 1e-6, 1e-5]),
            ]
        )

        dataset_precached = PersistentDataset(data=test_data, transform=test_transform, cache_dir=tempdir)
        data1_precached = dataset_precached[0]
        data2_precached = dataset_precached[1]

        dataset_postcached = PersistentDataset(data=test_data, transform=test_transform, cache_dir=tempdir)
        data1_postcached = dataset_postcached[0]
        data2_postcached = dataset_postcached[1]
        shutil.rmtree(tempdir)

        self.assertTupleEqual(data1_precached["image"].shape, expected_shape)
        self.assertTupleEqual(data1_precached["label"].shape, expected_shape)
        self.assertTupleEqual(data1_precached["extra"].shape, expected_shape)
        self.assertTupleEqual(data2_precached["image"].shape, expected_shape)
        self.assertTupleEqual(data2_precached["label"].shape, expected_shape)
        self.assertTupleEqual(data2_precached["extra"].shape, expected_shape)

        self.assertTupleEqual(data1_postcached["image"].shape, expected_shape)
        self.assertTupleEqual(data1_postcached["label"].shape, expected_shape)
        self.assertTupleEqual(data1_postcached["extra"].shape, expected_shape)
        self.assertTupleEqual(data2_postcached["image"].shape, expected_shape)
        self.assertTupleEqual(data2_postcached["label"].shape, expected_shape)
        self.assertTupleEqual(data2_postcached["extra"].shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
