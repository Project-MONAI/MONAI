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
from parameterized import parameterized

from monai.data import write_nifti
from monai.transforms import LoadImage, Orientation, Spacing
from tests.utils import make_nifti_image

TEST_IMAGE = np.arange(24).reshape((2, 4, 3))
TEST_AFFINE = np.array(
    [[-5.3, 0.0, 0.0, 102.01], [0.0, 0.52, 2.17, -7.50], [-0.0, 1.98, -0.26, -23.12], [0.0, 0.0, 0.0, 1.0]]
)

TEST_CASES = [
    [
        TEST_IMAGE,
        TEST_AFFINE,
        dict(reader="NibabelReader", image_only=False, as_closest_canonical=True),
        np.arange(24).reshape((2, 4, 3)),
    ],
    [
        TEST_IMAGE,
        TEST_AFFINE,
        dict(reader="NibabelReader", image_only=True, as_closest_canonical=True),
        np.array(
            [
                [[12.0, 15.0, 18.0, 21.0], [13.0, 16.0, 19.0, 22.0], [14.0, 17.0, 20.0, 23.0]],
                [[0.0, 3.0, 6.0, 9.0], [1.0, 4.0, 7.0, 10.0], [2.0, 5.0, 8.0, 11.0]],
            ]
        ),
    ],
    [
        TEST_IMAGE,
        TEST_AFFINE,
        dict(reader="NibabelReader", image_only=True, as_closest_canonical=False),
        np.arange(24).reshape((2, 4, 3)),
    ],
    [
        TEST_IMAGE,
        TEST_AFFINE,
        dict(reader="NibabelReader", image_only=False, as_closest_canonical=False),
        np.arange(24).reshape((2, 4, 3)),
    ],
    [
        TEST_IMAGE,
        None,
        dict(reader="NibabelReader", image_only=False, as_closest_canonical=False),
        np.arange(24).reshape((2, 4, 3)),
    ],
]


class TestNiftiLoadRead(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test_orientation(self, array, affine, reader_param, expected):
        test_image = make_nifti_image(array, affine)

        # read test cases
        loader = LoadImage(**reader_param)
        load_result = loader(test_image)
        if isinstance(load_result, tuple):
            data_array, header = load_result
        else:
            data_array = load_result
            header = None
        if os.path.exists(test_image):
            os.remove(test_image)

        # write test cases
        if header is not None:
            write_nifti(data_array, test_image, header["affine"], header.get("original_affine", None))
        elif affine is not None:
            write_nifti(data_array, test_image, affine)
        saved = nib.load(test_image)
        saved_affine = saved.affine
        saved_data = saved.get_fdata()
        if os.path.exists(test_image):
            os.remove(test_image)

        if affine is not None:
            np.testing.assert_allclose(saved_affine, affine)
        np.testing.assert_allclose(saved_data, expected)

    def test_consistency(self):
        np.set_printoptions(suppress=True, precision=3)
        test_image = make_nifti_image(np.arange(64).reshape(1, 8, 8), np.diag([1.5, 1.5, 1.5, 1]))
        data, header = LoadImage(reader="NibabelReader", as_closest_canonical=False)(test_image)
        data, original_affine, new_affine = Spacing([0.8, 0.8, 0.8])(data[None], header["affine"], mode="nearest")
        data, _, new_affine = Orientation("ILP")(data, new_affine)
        if os.path.exists(test_image):
            os.remove(test_image)
        write_nifti(data[0], test_image, new_affine, original_affine, mode="nearest", padding_mode="border")
        saved = nib.load(test_image)
        saved_data = saved.get_fdata()
        np.testing.assert_allclose(saved_data, np.arange(64).reshape(1, 8, 8), atol=1e-7)
        if os.path.exists(test_image):
            os.remove(test_image)
        write_nifti(
            data[0],
            test_image,
            new_affine,
            original_affine,
            mode="nearest",
            padding_mode="border",
            output_spatial_shape=(1, 8, 8),
        )
        saved = nib.load(test_image)
        saved_data = saved.get_fdata()
        np.testing.assert_allclose(saved_data, np.arange(64).reshape(1, 8, 8), atol=1e-7)
        if os.path.exists(test_image):
            os.remove(test_image)
        # test the case that only correct orientation but don't resample
        write_nifti(data[0], test_image, new_affine, original_affine, resample=False)
        saved = nib.load(test_image)
        # compute expected affine
        start_ornt = nib.orientations.io_orientation(new_affine)
        target_ornt = nib.orientations.io_orientation(original_affine)
        ornt_transform = nib.orientations.ornt_transform(start_ornt, target_ornt)
        data_shape = data[0].shape
        expected_affine = new_affine @ nib.orientations.inv_ornt_aff(ornt_transform, data_shape)
        np.testing.assert_allclose(saved.affine, expected_affine)
        if os.path.exists(test_image):
            os.remove(test_image)

    def test_write_2d(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.nii.gz")
            img = np.arange(6).reshape((2, 3))
            write_nifti(img, image_name, affine=np.diag([1]), target_affine=np.diag([1.4]))
            out = nib.load(image_name)
            np.testing.assert_allclose(out.get_fdata(), [[0, 1, 2], [3.0, 4, 5]])
            np.testing.assert_allclose(out.affine, np.diag([1.4, 1, 1, 1]))

            image_name = os.path.join(out_dir, "test1.nii.gz")
            img = np.arange(5).reshape((1, 5))
            write_nifti(img, image_name, affine=np.diag([1, 1, 1, 3, 3]), target_affine=np.diag([1.4, 2.0, 1, 3, 5]))
            out = nib.load(image_name)
            np.testing.assert_allclose(out.get_fdata(), [[0, 2, 4]])
            np.testing.assert_allclose(out.affine, np.diag([1.4, 2, 1, 1]))

    def test_write_3d(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.nii.gz")
            img = np.arange(6).reshape((1, 2, 3))
            write_nifti(img, image_name, affine=np.diag([1]), target_affine=np.diag([1.4]))
            out = nib.load(image_name)
            np.testing.assert_allclose(out.get_fdata(), [[[0, 1, 2], [3, 4, 5]]])
            np.testing.assert_allclose(out.affine, np.diag([1.4, 1, 1, 1]))

            image_name = os.path.join(out_dir, "test1.nii.gz")
            img = np.arange(5).reshape((1, 1, 5))
            write_nifti(img, image_name, affine=np.diag([1, 1, 1, 3, 3]), target_affine=np.diag([1.4, 2.0, 2, 3, 5]))
            out = nib.load(image_name)
            np.testing.assert_allclose(out.get_fdata(), [[[0, 2, 4]]])
            np.testing.assert_allclose(out.affine, np.diag([1.4, 2, 2, 1]))

    def test_write_4d(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.nii.gz")
            img = np.arange(6).reshape((1, 1, 3, 2))
            write_nifti(img, image_name, affine=np.diag([1.4, 1]), target_affine=np.diag([1, 1.4, 1]))
            out = nib.load(image_name)
            np.testing.assert_allclose(out.get_fdata(), [[[[0, 1], [2, 3], [4, 5]]]])
            np.testing.assert_allclose(out.affine, np.diag([1, 1.4, 1, 1]))

            image_name = os.path.join(out_dir, "test1.nii.gz")
            img = np.arange(5).reshape((1, 1, 5, 1))
            write_nifti(img, image_name, affine=np.diag([1, 1, 1, 3, 3]), target_affine=np.diag([1.4, 2.0, 2, 3, 5]))
            out = nib.load(image_name)
            np.testing.assert_allclose(out.get_fdata(), [[[[0], [2], [4]]]])
            np.testing.assert_allclose(out.affine, np.diag([1.4, 2, 2, 1]))

    def test_write_5d(self):
        with tempfile.TemporaryDirectory() as out_dir:
            image_name = os.path.join(out_dir, "test.nii.gz")
            img = np.arange(12).reshape((1, 1, 3, 2, 2))
            write_nifti(img, image_name, affine=np.diag([1]), target_affine=np.diag([1.4]))
            out = nib.load(image_name)
            np.testing.assert_allclose(
                out.get_fdata(),
                np.array([[[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]], [[8.0, 9.0], [10.0, 11.0]]]]]),
            )
            np.testing.assert_allclose(out.affine, np.diag([1.4, 1, 1, 1]))

            image_name = os.path.join(out_dir, "test1.nii.gz")
            img = np.arange(10).reshape((1, 1, 5, 1, 2))
            write_nifti(img, image_name, affine=np.diag([1, 1, 1, 3, 3]), target_affine=np.diag([1.4, 2.0, 2, 3, 5]))
            out = nib.load(image_name)
            np.testing.assert_allclose(out.get_fdata(), np.array([[[[[0.0, 1.0]], [[4.0, 5.0]], [[8.0, 9.0]]]]]))
            np.testing.assert_allclose(out.affine, np.diag([1.4, 2, 2, 1]))


if __name__ == "__main__":
    unittest.main()
