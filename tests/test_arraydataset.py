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
from monai.data import ArrayDataset
from monai.transforms import Compose, LoadNifti, AddChannel, RandAdjustContrast, Spacing

TEST_CASE_1 = [
    Compose([LoadNifti(image_only=True), AddChannel(), RandAdjustContrast()]),
    Compose([LoadNifti(image_only=True), AddChannel(), RandAdjustContrast()]),
    (0, 1),
    (1, 128, 128, 128),
]


class TestCompose(Compose):
    def __call__(self, input_):
        img, metadata = self.transforms[0](input_)
        img = self.transforms[1](img)
        img, _, _ = self.transforms[2](img, metadata["affine"])
        return self.transforms[3](img), metadata


TEST_CASE_2 = [
    TestCompose([LoadNifti(image_only=False), AddChannel(), Spacing(pixdim=(2, 2, 4)), RandAdjustContrast()]),
    TestCompose([LoadNifti(image_only=False), AddChannel(), Spacing(pixdim=(2, 2, 4)), RandAdjustContrast()]),
    (0, 2),
    (1, 64, 64, 33),
]


class TestArrayDataset(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, img_transform, label_transform, indexes, expected_shape):
        test_image = nib.Nifti1Image(np.random.randint(0, 2, size=(128, 128, 128)), np.eye(4))
        tempdir = tempfile.mkdtemp()
        test_image1 = os.path.join(tempdir, "test_image1.nii.gz")
        test_seg1 = os.path.join(tempdir, "test_seg1.nii.gz")
        test_image2 = os.path.join(tempdir, "test_image2.nii.gz")
        test_seg2 = os.path.join(tempdir, "test_seg2.nii.gz")
        nib.save(test_image, test_image1)
        nib.save(test_image, test_seg1)
        nib.save(test_image, test_image2)
        nib.save(test_image, test_seg2)
        test_images = [test_image1, test_image2]
        test_segs = [test_seg1, test_seg2]
        test_labels = [1, 1]
        dataset = ArrayDataset(test_images, img_transform, test_segs, label_transform, test_labels, None)
        data1 = dataset[0]
        data2 = dataset[1]

        self.assertTupleEqual(data1[indexes[0]].shape, expected_shape)
        self.assertTupleEqual(data1[indexes[1]].shape, expected_shape)
        np.testing.assert_allclose(data1[indexes[0]], data1[indexes[1]])
        self.assertTupleEqual(data2[indexes[0]].shape, expected_shape)
        self.assertTupleEqual(data2[indexes[1]].shape, expected_shape)
        np.testing.assert_allclose(data2[indexes[0]], data2[indexes[0]])
        shutil.rmtree(tempdir)


if __name__ == "__main__":
    unittest.main()
