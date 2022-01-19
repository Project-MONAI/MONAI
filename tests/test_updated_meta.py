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

import numpy as np
from parameterized import parameterized

import monai.transforms as mt
from monai.data import create_test_image_3d
from monai.utils.enums import CommonKeys, PostFix
from tests.utils import make_nifti_image, make_rand_affine

MAX_NUM_VOXELS_TO_CROP = 10
IM_SHAPE = (100, 101, 107)
IM = CommonKeys.IMAGE
SPATIAL_CROP = mt.SpatialCropd(IM, roi_start=(MAX_NUM_VOXELS_TO_CROP, 0, 0), roi_end=IM_SHAPE)
CROP_FOREGROUND = mt.CropForegroundd(IM, IM)
MAX_DIFF = 3  # max allowed difference is 3 (both voxels and mm)

TESTS = []
for trans in (SPATIAL_CROP, CROP_FOREGROUND):
    for resample in (False, True):
        TESTS.append((trans, resample))


class TestUpdatedMeta(unittest.TestCase):
    def setUp(self):
        affine = make_rand_affine()
        # affine = np.eye(4)
        im, _ = create_test_image_3d(*IM_SHAPE)
        # we want cropping to be lossless, so ensure the border is greater than the amount to be cropped.
        im = mt.BorderPad(2 * MAX_NUM_VOXELS_TO_CROP)(im)
        monai_data = os.environ.get("MONAI_DATA_DIRECTORY") or tempfile.mkdtemp()
        self.folder = os.path.join(monai_data, "test_updated_meta")
        self.im_fname = make_nifti_image(im, affine, dir=self.folder, fname="orig", suffix=".nii", verbose=True)
        self.loader = mt.Compose([mt.LoadImaged(IM), mt.AddChanneld(IM)])

    @staticmethod
    def get_center_mass(data, key, select_fn=lambda x: x > 0):
        # get center of mass of all voxels that conform to select_fn
        img = data[key]
        if img.ndim != 4 or img.shape[0] != 1:
            raise RuntimeError("Expect shape to be `CHWD` where `C==1`.")
        img = select_fn(img[0])  # remove channel dim
        affine = data[PostFix.meta(key)]["affine"]
        com_img = [np.average(i) for i in np.where(img)]
        com_real = affine @ (com_img + [1])
        com_real = com_real[:-1]
        return np.asarray(com_img), com_real

    @staticmethod
    def get_diff_com(com1, com2):
        # Euclidean distance between two COMs
        return np.sum((com1 - com2) ** 2) ** 0.5

    def check_coms(self, com1_img, com1_real, com2_img, com2_real, img_should_match=False):
        # check that we've sufficiently modified the image...
        check_img_space = self.assertLess if img_should_match else self.assertGreater
        check_img_space(self.get_diff_com(com1_img, com2_img), MAX_DIFF)
        # but correctly updated the real-world coordinates
        self.assertLess(self.get_diff_com(com1_real, com2_real), MAX_DIFF)

    @parameterized.expand(TESTS)
    def test_updated_meta(self, extra_trans, resample):
        d = self.loader({IM: self.im_fname})
        coms1 = self.get_center_mass(d, IM)
        d = extra_trans(d)
        coms2 = self.get_center_mass(d, IM)
        self.check_coms(*coms1, *coms2)

        # check that saving to file has the same result
        prefix = "spatial_crop" if isinstance(extra_trans, mt.SpatialCropd) else "crop_foreground"
        if resample:
            prefix += "_resample"

        # TODO: remove this line
        del d[PostFix.meta(IM)]["original_affine"]

        saver = mt.SaveImaged(
            IM,
            output_dir=self.folder,
            output_ext=".nii",
            resample=resample,
            separate_folder=False,
            padding_mode="zeros",
            output_postfix=prefix,
        )
        d = saver(d)
        saved_fname = os.path.join(self.folder, self.im_fname[:-4] + "_" + prefix + ".nii")
        d = self.loader({IM: saved_fname})
        coms2 = self.get_center_mass(d, IM)
        # if output image is resampled, imgs should also match in image space
        img_should_match = resample and "original_affine" in d
        self.check_coms(*coms1, *coms2, img_should_match)


if __name__ == "__main__":
    unittest.main()
