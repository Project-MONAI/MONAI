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
import tempfile
import unittest
from copy import deepcopy

import numpy as np
from parameterized import parameterized

import monai.transforms as mt
from monai.data import create_test_image_3d
from monai.utils.enums import CommonKeys, PostFix
from tests.utils import make_nifti_image, make_rand_affine

IM_SHAPE = (100, 101, 107)
IM = CommonKeys.IMAGE
SPATIAL_CROP = mt.SpatialCropd(IM, roi_start=(10, 3, 7), roi_end=IM_SHAPE)
CROP_FOREGROUND = mt.CropForegroundd(IM, IM)
MAX_DIFF = 3  # max allowed difference (both voxels and mm)

TESTS = []
for trans in (SPATIAL_CROP, CROP_FOREGROUND):
    for resample in (False, True):
        TESTS.append((trans, resample))


class TestUpdatedMeta(unittest.TestCase):
    def setUp(self):
        affine = make_rand_affine()
        _, im = create_test_image_3d(*IM_SHAPE)
        monai_data = os.environ.get("MONAI_DATA_DIRECTORY") or tempfile.mkdtemp()
        self.folder = os.path.join(monai_data, "test_updated_meta")
        self.im_fname = make_nifti_image(im, affine, dir=self.folder, fname="orig", verbose=True)
        self.loader = mt.Compose([mt.LoadImaged(IM), mt.AddChanneld(IM)])

    @staticmethod
    def get_center_mass(data, key, select_fn=lambda x: x > 0):
        """
        Get center of mass of all voxels that conform to select_fn.
        Return COM in both image and real space.
        """
        img = data[key]
        if img.ndim != 4 or img.shape[0] != 1:
            raise RuntimeError("Expect shape to be `CHWD` where `C==1`.")
        img = select_fn(img[0])  # remove channel dim and convert to binary mask
        affine = data[PostFix.meta(key)]["affine"]
        com_img = [np.average(i) for i in np.where(img)]
        com_real = affine @ (com_img + [1])
        com_real = com_real[:-1]
        return np.asarray(com_img), com_real

    @staticmethod
    def get_diff_com(com1, com2):
        """Euclidean distance between two COMs."""
        return np.sum((com1 - com2) ** 2) ** 0.5

    def check_coms(self, d1, d2, img_should_match=False):
        """
        Check that COMs match in real space.
        If image has not been resampled, there should be a discrepency in image space.
        """
        com1_img, com1_real = self.get_center_mass(d1, IM)
        com2_img, com2_real = self.get_center_mass(d2, IM)
        com_diff_img = self.get_diff_com(com1_img, com2_img)
        com_diff_real = self.get_diff_com(com1_real, com2_real)

        # if images have been resampled, then they should match in image space. else, not.
        if img_should_match:
            self.assertLess(com_diff_img, MAX_DIFF)
        else:
            self.assertGreater(com_diff_img, MAX_DIFF)
        # either way, check they match in real space.
        self.assertLess(com_diff_real, MAX_DIFF)

    @parameterized.expand(TESTS)
    def test_updated_meta(self, extra_trans, resample):
        d_in = self.loader({IM: self.im_fname})
        d_out = extra_trans(deepcopy(d_in))
        self.check_coms(d_in, d_out)

        # check that saving to file has the same result
        prefix = "spatial_crop" if isinstance(extra_trans, mt.SpatialCropd) else "crop_foreground"
        if resample:
            prefix += "_resample"

        # TODO: remove this line
        if resample:
            del d_out[PostFix.meta(IM)]["original_affine"]

        saver = mt.SaveImaged(
            IM,
            output_dir=self.folder,
            resample=resample,
            separate_folder=False,
            padding_mode="zeros",
            output_postfix=prefix,
        )
        d_out = saver(d_out)
        saved_fname = os.path.join(self.folder, self.im_fname[:-7] + "_" + prefix + ".nii.gz")
        d_out = self.loader({IM: saved_fname})
        # if output image is resampled, imgs should also match in image space
        img_should_match = resample and "original_affine" in d_out
        self.check_coms(d_in, d_out, img_should_match)


if __name__ == "__main__":
    unittest.main()
