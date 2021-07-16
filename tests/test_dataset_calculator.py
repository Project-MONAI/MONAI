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

import glob
import os
import tempfile
import unittest

import nibabel as nib
import numpy as np

from monai.data import DatasetCalculator, create_test_image_3d
from monai.utils import set_determinism


class TestDatasetCalculator(unittest.TestCase):
    def test_spacing_intensity(self):
        set_determinism(seed=0)
        with tempfile.TemporaryDirectory() as tempdir:

            for i in range(5):
                im, seg = create_test_image_3d(32, 32, 32, num_seg_classes=1, num_objs=3, rad_max=6, channel_dim=-1)
                n = nib.Nifti1Image(im, np.eye(4))
                nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))
                n = nib.Nifti1Image(seg, np.eye(4))
                nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))

            train_images = sorted(glob.glob(os.path.join(tempdir, "img*.nii.gz")))
            train_labels = sorted(glob.glob(os.path.join(tempdir, "seg*.nii.gz")))
            data_dicts = [
                {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
            ]

            calculator = DatasetCalculator(data_dicts)
            target_spacing = calculator._get_target_spacing(anisotropic_threshold=3, percentile=10.0)
            self.assertEqual(target_spacing, (1.0, 1.0, 1.0))
            intensity_stats = calculator._get_intensity_stats(lower=0.5, upper=99.5)
            self.assertEqual(intensity_stats, (0.56, 1.0, 0.89, 0.13))


if __name__ == "__main__":
    unittest.main()
