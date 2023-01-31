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

from __future__ import annotations

import os
import unittest

import numpy as np
import torch
from parameterized import parameterized

from monai.apps import download_url
from monai.data import ITKReader
from monai.data.itk_torch_affine_matrix_bridge import (
    create_itk_affine_from_parameters,
    itk_affine_resample,
    itk_image_to_metatensor,
    itk_to_monai_affine,
    itk_warp,
    monai_affine_resample,
    monai_to_itk_affine,
    monai_warp,
    remove_border,
)
from monai.utils import optional_import
from tests.utils import skip_if_downloading_fails, testing_data_config

itk, has_itk = optional_import("itk")

TESTS = ["CT_2D_head_fixed.mha", "CT_2D_head_moving.mha", "copd1_highres_INSP_STD_COPD_img.nii.gz"]

key = "copd1_highres_INSP_STD_COPD_img"
FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", f"{key}.nii.gz")


@unittest.skipUnless(has_itk, "Requires `itk` package.")
class TestITKTorchAffineMatrixBridge(unittest.TestCase):
    def setUp(self):
        # TODO: which data should be used
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        self.reader = ITKReader()

        for k, n in ((key, FILE_PATH),):
            if not os.path.exists(n):
                with skip_if_downloading_fails():
                    data_spec = testing_data_config("images", f"{k}")
                    download_url(data_spec["url"], n, hash_val=data_spec["hash_val"], hash_type=data_spec["hash_type"])

    @parameterized.expand(TESTS)
    def test_setting_affine_parameters(self, filepath):
        # Read image
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = remove_border(image)
        ndim = image.ndim

        # Affine parameters
        translation = [65.2, -50.2, 33.9][:ndim]
        rotation = [0.78539816339, 1.0, -0.66][:ndim]
        scale = [2.0, 1.5, 3.2][:ndim]
        shear = [0, 1, 1.6]  # axis1, axis2, coeff

        # Spacing
        spacing = np.array([1.2, 1.5, 2.0])[:ndim]
        image.SetSpacing(spacing)

        # ITK
        matrix, translation = create_itk_affine_from_parameters(image, translation, rotation, scale, shear)
        output_array_itk = itk_affine_resample(image, matrix=matrix, translation=translation)

        # MONAI
        metatensor = itk_image_to_metatensor(image)
        affine_matrix_for_monai = itk_to_monai_affine(image, matrix, translation)
        output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

        ###########################################################################
        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor.squeeze().permute(*torch.arange(metatensor.ndim - 2, -1, -1)).array
        np.testing.assert_array_equal(input_array_monai, np.asarray(image))

        # Compare outputs
        percentage = (
            100 * np.isclose(output_array_monai, output_array_itk).sum(dtype=np.float64) / output_array_itk.size
        )
        print("MONAI equals result: ", percentage, "%")
        self.assertGreaterEqual(percentage, 99.0)

        diff_output = output_array_monai - output_array_itk
        print(f"[Min, Max] MONAI: [{output_array_monai.min()}, {output_array_monai.max()}]")
        print(f"[Min, Max] ITK: [{output_array_itk.min()}, {output_array_itk.max()}]")
        print(f"[Min, Max] diff: [{diff_output.min()}, {diff_output.max()}]")

        # Write
        # itk.imwrite(itk.GetImageFromArray(diff_output), "./output/diff.tif")
        # itk.imwrite(itk.GetImageFromArray(output_array_monai), "./output/output_monai.tif")
        # itk.imwrite(itk.GetImageFromArray(output_array_itk), "./output/output_itk.tif")
        ###########################################################################

    @parameterized.expand(TESTS)
    def test_arbitary_center_of_rotation(self, filepath):
        # Read image
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = remove_border(image)
        ndim = image.ndim

        # ITK matrix (3x3 affine matrix)
        matrix = np.array(
            [
                [0.55915995, 0.50344867, 0.43208387],
                [0.01133669, 0.82088571, 0.86841365],
                [0.30478496, 0.94998986, 0.32742505],
            ]
        )[:ndim, :ndim]
        translation = [54.0, 2.7, -11.9][:ndim]

        # Spatial properties
        center_of_rotation = [-32.3, 125.1, 0.7][:ndim]
        origin = [1.6, 0.5, 2.0][:ndim]
        spacing = np.array([1.2, 1.5, 0.6])[:ndim]

        image.SetSpacing(spacing)
        image.SetOrigin(origin)

        # ITK
        output_array_itk = itk_affine_resample(image, matrix, translation, center_of_rotation)

        # MONAI
        metatensor = itk_image_to_metatensor(image)
        affine_matrix_for_monai = itk_to_monai_affine(image, matrix, translation, center_of_rotation)
        output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor.squeeze().permute(*torch.arange(metatensor.ndim - 2, -1, -1)).array
        np.testing.assert_array_equal(input_array_monai, np.asarray(image))

        ###########################################################################
        # Compare outputs
        percentage = (
            100 * np.isclose(output_array_monai, output_array_itk).sum(dtype=np.float64) / output_array_itk.size
        )
        print("MONAI equals result: ", percentage, "%")
        self.assertGreaterEqual(percentage, 99.0)

        diff_output = output_array_monai - output_array_itk
        print(f"[Min, Max] MONAI: [{output_array_monai.min()}, {output_array_monai.max()}]")
        print(f"[Min, Max] ITK: [{output_array_itk.min()}, {output_array_itk.max()}]")
        print(f"[Min, Max] diff: [{diff_output.min()}, {diff_output.max()}]")
        ###########################################################################

    @parameterized.expand(TESTS)
    def test_monai_to_itk(self, filepath):
        # Read image
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = remove_border(image)
        ndim = image.ndim

        # MONAI affine matrix
        affine_matrix = torch.eye(ndim + 1, dtype=torch.float64)
        affine_matrix[:ndim, :ndim] = torch.tensor(
            [
                [0.55915995, 0.50344867, 0.43208387],
                [0.01133669, 0.82088571, 0.86841365],
                [0.30478496, 0.94998986, 0.32742505],
            ],
            dtype=torch.float64,
        )[:ndim, :ndim]

        affine_matrix[:ndim, ndim] = torch.tensor([54.0, 2.7, -11.9], dtype=torch.float64)[:ndim]

        # Spatial properties
        center_of_rotation = [-32.3, 125.1, 0.7][:ndim]
        origin = [1.6, 0.5, 2.0][:ndim]
        spacing = np.array([1.2, 1.5, 0.6])[:ndim]

        image.SetSpacing(spacing)
        image.SetOrigin(origin)

        # ITK
        matrix, translation = monai_to_itk_affine(image, affine_matrix, center_of_rotation)
        output_array_itk = itk_affine_resample(image, matrix, translation, center_of_rotation)

        # MONAI
        metatensor = itk_image_to_metatensor(image)
        output_array_monai = monai_affine_resample(metatensor, affine_matrix)

        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor.squeeze().permute(*torch.arange(metatensor.ndim - 2, -1, -1)).array
        np.testing.assert_array_equal(input_array_monai, np.asarray(image))

        ###########################################################################
        # Compare outputs
        percentage = (
            100 * np.isclose(output_array_monai, output_array_itk).sum(dtype=np.float64) / output_array_itk.size
        )
        print("MONAI equals result: ", percentage, "%")
        self.assertGreaterEqual(percentage, 99.0)

        diff_output = output_array_monai - output_array_itk
        print(f"[Min, Max] MONAI: [{output_array_monai.min()}, {output_array_monai.max()}]")
        print(f"[Min, Max] ITK: [{output_array_itk.min()}, {output_array_itk.max()}]")
        print(f"[Min, Max] diff: [{diff_output.min()}, {diff_output.max()}]")
        ###########################################################################

    @parameterized.expand(TESTS)
    def test_cyclic_conversion(self, filepath):
        image = self.reader.read(os.path.join(self.data_dir, filepath))
        image[:] = remove_border(image)
        ndim = image.ndim

        # ITK matrix (3x3 affine matrix)
        matrix = np.array(
            [
                [2.90971094, 1.18297296, 2.60008784],
                [0.29416137, 0.10294283, 2.82302616],
                [1.70578374, 1.39706003, 2.54652029],
            ]
        )[:ndim, :ndim]

        translation = [-29.05463245, 35.27116398, 48.58759597][:ndim]

        # Spatial properties
        center_of_rotation = [-27.84789587, -60.7871084, 42.73501932][:ndim]
        origin = [8.10416794, 5.4831944, 0.49211025][:ndim]
        spacing = np.array([0.7, 3.2, 1.3])[:ndim]

        image.SetSpacing(spacing)
        image.SetOrigin(origin)

        affine_matrix = itk_to_monai_affine(image, matrix, translation, center_of_rotation)

        matrix_result, translation_result = monai_to_itk_affine(image, affine_matrix, center_of_rotation)

        np.testing.assert_allclose(matrix, matrix_result)
        np.testing.assert_allclose(translation, translation_result)

    @parameterized.expand([(2,), (3,)])
    def test_random_array(self, ndim):
        # Create image/array with random size and pixel intensities
        s = torch.randint(low=2, high=20, size=(ndim,))
        img = 100 * torch.rand((1, 1, *s.tolist()), dtype=torch.float32)

        # Pad at the edges because ITK and MONAI have different behavior there
        # during resampling
        img = torch.nn.functional.pad(img, pad=ndim * (1, 1))
        ddf = 5 * torch.rand((1, ndim, *img.shape[-ndim:]), dtype=torch.float32) - 2.5

        # Warp with MONAI
        img_resampled = monai_warp(img, ddf)

        # Create ITK image
        itk_img = itk.GetImageFromArray(img.squeeze().numpy())

        # Set random spacing
        spacing = 3 * np.random.rand(ndim)
        itk_img.SetSpacing(spacing)

        # Set random direction
        direction = 5 * np.random.rand(ndim, ndim) - 5
        direction = itk.matrix_from_array(direction)
        itk_img.SetDirection(direction)

        # Set random origin
        origin = 100 * np.random.rand(ndim) - 100
        itk_img.SetOrigin(origin)

        # Warp with ITK
        itk_img_resampled = itk_warp(itk_img, ddf.squeeze().numpy())

        # Compare
        np.testing.assert_allclose(img_resampled, itk_img_resampled, rtol=1e-3, atol=1e-3)
        diff_output = img_resampled - itk_img_resampled
        print(f"[Min, Max] diff: [{diff_output.min()}, {diff_output.max()}]")

    @parameterized.expand(TESTS)
    def test_real_data(self, filepath):
        # Read image
        image = itk.imread(os.path.join(self.data_dir, filepath), itk.F)
        image[:] = remove_border(image)
        ndim = image.ndim

        # Random ddf
        ddf = 10 * torch.rand((1, ndim, *image.shape), dtype=torch.float32) - 10

        # Warp with MONAI
        image_tensor = torch.tensor(itk.GetArrayFromImage(image), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        img_resampled = monai_warp(image_tensor, ddf)

        # Warp with ITK
        itk_img_resampled = itk_warp(image, ddf.squeeze().numpy())

        # Compare
        np.testing.assert_allclose(img_resampled, itk_img_resampled, rtol=1e-3, atol=1e-3)
        diff_output = img_resampled - itk_img_resampled
        print(f"[Min, Max] diff: [{diff_output.min()}, {diff_output.max()}]")


if __name__ == "__main__":
    unittest.main()
