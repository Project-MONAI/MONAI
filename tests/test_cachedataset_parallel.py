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
from monai.data import CacheDataset
from monai.transforms import Compose, LoadNiftid

TEST_CASE_1 = [0, 100]

TEST_CASE_2 = [4, 100]


class TestCacheDatasetParallel(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, num_workers, dataset_size):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=[128, 128, 128]), np.eye(4))
        tempdir = tempfile.mkdtemp()
        nib.save(test_image, os.path.join(tempdir, "test_image1.nii.gz"))
        nib.save(test_image, os.path.join(tempdir, "test_label1.nii.gz"))
        nib.save(test_image, os.path.join(tempdir, "test_extra1.nii.gz"))
        test_data = [
            {
                "image": os.path.join(tempdir, "test_image1.nii.gz"),
                "label": os.path.join(tempdir, "test_label1.nii.gz"),
                "extra": os.path.join(tempdir, "test_extra1.nii.gz"),
            }
        ] * dataset_size
        dataset = CacheDataset(
            data=test_data,
            transform=Compose([LoadNiftid(keys=["image", "label", "extra"])]),
            cache_rate=1,
            num_workers=num_workers,
        )
        shutil.rmtree(tempdir)
        self.assertEqual(len(dataset._cache), dataset.cache_num)
        for i in range(dataset.cache_num):
            self.assertIsNotNone(dataset._cache[i])


if __name__ == "__main__":
    unittest.main()
