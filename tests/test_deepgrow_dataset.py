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
import tempfile
import unittest

import nibabel as nib
import numpy as np

from monai.apps.deepgrow.dataset import create_dataset


class TestCreateDataset(unittest.TestCase):
    def _create_data(self, tempdir):
        image = np.random.randint(0, 2, size=(4, 4, 4))
        image_file = os.path.join(tempdir, "image1.nii.gz")
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_file)

        label = np.random.randint(0, 1, size=(4, 4, 4))
        label[0][0][2] = 1
        label[0][1][2] = 1
        label[0][1][0] = 1
        label_file = os.path.join(tempdir, "label1.nii.gz")
        nib.save(nib.Nifti1Image(label, np.eye(4)), label_file)

        return [{"image": image_file, "label": label_file}]

    def test_create_dataset_2d(self):
        with tempfile.TemporaryDirectory() as tempdir:
            datalist = self._create_data(tempdir)
            output_dir = os.path.join(tempdir, "2d")
            deepgrow_datalist = create_dataset(datalist=datalist, output_dir=output_dir, dimension=2, pixdim=(1, 1))
            assert len(deepgrow_datalist) == 2 and deepgrow_datalist[0]["region"] == 1

    def test_create_dataset_3d(self):
        with tempfile.TemporaryDirectory() as tempdir:
            datalist = self._create_data(tempdir)
            output_dir = os.path.join(tempdir, "3d")
            deepgrow_datalist = create_dataset(datalist=datalist, output_dir=output_dir, dimension=3, pixdim=(1, 1, 1))
            assert len(deepgrow_datalist) == 1 and deepgrow_datalist[0]["region"] == 1


if __name__ == "__main__":
    unittest.main()
