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
from pathlib import Path

import itk
import nibabel as nib
import numpy as np
from parameterized import parameterized

from monai.data import ITKReader
from monai.transforms import Compose, EnsureChannelFirstD, LoadImaged, SaveImageD
from monai.utils.enums import PostFix

KEYS = ["image", "label", "extra"]

TEST_CASE_1 = [{"keys": KEYS}, (128, 128, 128)]

TEST_CASE_2 = [{"keys": KEYS, "reader": "ITKReader", "fallback_only": False}, (128, 128, 128)]


class TestLoadImaged(unittest.TestCase):
    @parameterized.expand([TEST_CASE_1, TEST_CASE_2])
    def test_shape(self, input_param, expected_shape):
        test_image = nib.Nifti1Image(np.random.rand(128, 128, 128), np.eye(4))
        test_data = {}
        with tempfile.TemporaryDirectory() as tempdir:
            for key in KEYS:
                nib.save(test_image, os.path.join(tempdir, key + ".nii.gz"))
                test_data.update({key: os.path.join(tempdir, key + ".nii.gz")})
            result = LoadImaged(**input_param)(test_data)

        for key in KEYS:
            self.assertTupleEqual(result[key].shape, expected_shape)

    def test_register(self):
        spatial_size = (32, 64, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            itk_np_view = itk.image_view_from_array(test_image)
            itk.imwrite(itk_np_view, filename)

            loader = LoadImaged(keys="img")
            loader.register(ITKReader())
            result = loader({"img": Path(filename)})
            self.assertTupleEqual(tuple(result[PostFix.meta("img")]["spatial_shape"]), spatial_size[::-1])
            self.assertTupleEqual(result["img"].shape, spatial_size[::-1])

    def test_channel_dim(self):
        spatial_size = (32, 64, 3, 128)
        test_image = np.random.rand(*spatial_size)
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.nii.gz")
            nib.save(nib.Nifti1Image(test_image, affine=np.eye(4)), filename)

            loader = LoadImaged(keys="img")
            loader.register(ITKReader(channel_dim=2))
            result = EnsureChannelFirstD("img")(loader({"img": filename}))
            self.assertTupleEqual(tuple(result[PostFix.meta("img")]["spatial_shape"]), (32, 64, 128))
            self.assertTupleEqual(result["img"].shape, (3, 32, 64, 128))

    def test_no_file(self):
        with self.assertRaises(RuntimeError):
            LoadImaged(keys="img")({"img": "unknown"})
        with self.assertRaises(RuntimeError):
            LoadImaged(keys="img", reader="nibabelreader")({"img": "unknown"})


class TestConsistency(unittest.TestCase):
    def _cmp(self, filename, shape, ch_shape, reader_1, reader_2, outname, ext):
        data_dict = {"img": filename}
        keys = data_dict.keys()
        xforms = Compose([LoadImaged(keys, reader=reader_1, ensure_channel_first=True)])
        img_dict = xforms(data_dict)  # load dicom with itk
        self.assertTupleEqual(img_dict["img"].shape, ch_shape)
        self.assertTupleEqual(tuple(img_dict[PostFix.meta("img")]["spatial_shape"]), shape)

        with tempfile.TemporaryDirectory() as tempdir:
            save_xform = SaveImageD(
                keys, meta_keys=PostFix.meta("img"), output_dir=tempdir, squeeze_end_dims=False, output_ext=ext
            )
            save_xform(img_dict)  # save to nifti

            new_xforms = Compose([LoadImaged(keys, reader=reader_2), EnsureChannelFirstD(keys)])
            out = new_xforms({"img": os.path.join(tempdir, outname)})  # load nifti with itk
            self.assertTupleEqual(out["img"].shape, ch_shape)
            self.assertTupleEqual(tuple(out[PostFix.meta("img")]["spatial_shape"]), shape)
            if "affine" in img_dict[PostFix.meta("img")] and "affine" in out[PostFix.meta("img")]:
                np.testing.assert_allclose(
                    img_dict[PostFix.meta("img")]["affine"], out[PostFix.meta("img")]["affine"], rtol=1e-3
                )
            np.testing.assert_allclose(out["img"], img_dict["img"], rtol=1e-3)

    def test_dicom(self):
        img_dir = "tests/testing_data/CT_DICOM"
        self._cmp(
            img_dir, (16, 16, 4), (1, 16, 16, 4), "itkreader", "itkreader", "CT_DICOM/CT_DICOM_trans.nii.gz", ".nii.gz"
        )
        output_name = "CT_DICOM/CT_DICOM_trans.nii.gz"
        self._cmp(img_dir, (16, 16, 4), (1, 16, 16, 4), "nibabelreader", "itkreader", output_name, ".nii.gz")
        self._cmp(img_dir, (16, 16, 4), (1, 16, 16, 4), "itkreader", "nibabelreader", output_name, ".nii.gz")

    def test_multi_dicom(self):
        """multichannel dicom reading, saving to nifti, then load with itk or nibabel"""

        img_dir = ["tests/testing_data/CT_DICOM", "tests/testing_data/CT_DICOM"]
        self._cmp(
            img_dir, (16, 16, 4), (2, 16, 16, 4), "itkreader", "itkreader", "CT_DICOM/CT_DICOM_trans.nii.gz", ".nii.gz"
        )
        output_name = "CT_DICOM/CT_DICOM_trans.nii.gz"
        self._cmp(img_dir, (16, 16, 4), (2, 16, 16, 4), "nibabelreader", "itkreader", output_name, ".nii.gz")
        self._cmp(img_dir, (16, 16, 4), (2, 16, 16, 4), "itkreader", "nibabelreader", output_name, ".nii.gz")

    def test_png(self):
        """png reading with itk, saving to nifti, then load with itk or nibabel or PIL"""

        test_image = np.random.randint(0, 256, size=(256, 224, 3)).astype("uint8")
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.join(tempdir, "test_image.png")
            itk_np_view = itk.image_view_from_array(test_image, is_vector=True)
            itk.imwrite(itk_np_view, filename)
            output_name = "test_image/test_image_trans.png"
            self._cmp(filename, (224, 256), (3, 224, 256), "itkreader", "itkreader", output_name, ".png")
            self._cmp(filename, (224, 256), (3, 224, 256), "itkreader", "PILReader", output_name, ".png")
            self._cmp(filename, (224, 256), (3, 224, 256), "itkreader", "nibabelreader", output_name, ".png")


if __name__ == "__main__":
    unittest.main()
