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

from parameterized import parameterized

from monai.apps import download_and_extract
from monai.utils import optional_import
from tests.utils import skip_if_downloading_fails, testing_data_config

itk, has_itk = optional_import("itk")
import numpy as np
import torch

from monai.data.itk_torch_affine_matrix_bridge import (
    create_itk_affine_from_parameters,
    image_to_metatensor,
    itk_affine_resample,
    itk_to_monai_affine,
    metatensor_to_array,
    monai_affine_resample,
    monai_to_itk_affine,
    remove_border,
)

TEST_CASE_2D = {"filepath": "CT_2D_head_fixed.mha"}
# Download URL: https://data.kitware.com/api/v1/file/62a0f067bddec9d0c4175c5a/download
# SHA-521: 60193cd6ef0cf055c623046446b74f969a2be838444801bd32ad5bedc8a7eeecb343e8a1208769c9c7a711e101c806a3133eccdda7790c551a69a64b9b3701e9
TEST_CASE_3D = {"filepath": "copd1_highres_INSP_STD_COPD_img.nii.gz"}


@unittest.skipUnless(has_itk, "Requires `itk` package.")
class TestITKTorchAffineMatrixBridge(unittest.TestCase):
    def setUp(self):
        # TODO: which data should be used
        self.data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testing_data")
        data_dir = os.path.join(self.data_dir, "MedNIST")
        dataset_file = os.path.join(self.data_dir, "MedNIST.tar.gz")

        if not os.path.exists(data_dir):
            with skip_if_downloading_fails():
                data_spec = testing_data_config("images", "mednist")
                download_and_extract(
                    data_spec["url"],
                    dataset_file,
                    self.data_dir,
                    hash_val=data_spec["hash_val"],
                    hash_type=data_spec["hash_type"],
                )

    @parameterized.expand([TEST_CASE_2D, TEST_CASE_3D])
    def test_setting_affine_parameters(self, filepath):
        # Read image
        image = itk.imread(filepath, itk.F)
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
        matrix, translation = create_itk_affine_from_parameters(
            image, translation=translation, rotation=rotation, scale=scale, shear=shear
        )
        output_array_itk = itk_affine_resample(image, matrix=matrix, translation=translation)

        # MONAI
        metatensor = image_to_metatensor(image)
        affine_matrix_for_monai = itk_to_monai_affine(image, matrix=matrix, translation=translation)
        output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

        ###########################################################################
        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor_to_array(metatensor)
        assert np.array_equal(input_array_monai, np.asarray(image))

        # Compare outputs
        print("MONAI-ITK: ", np.allclose(output_array_monai, output_array_itk))

        diff_output = output_array_monai - output_array_itk
        print(f"[Min, Max] MONAI: [{output_array_monai.min()}, {output_array_monai.max()}]")
        print(f"[Min, Max] ITK: [{output_array_itk.min()}, {output_array_itk.max()}]")
        print(f"[Min, Max] diff: [{diff_output.min()}, {diff_output.max()}]")

        # Write
        # itk.imwrite(itk.GetImageFromArray(diff_output), "./output/diff.tif")
        # itk.imwrite(itk.GetImageFromArray(output_array_monai), "./output/output_monai.tif")
        # itk.imwrite(itk.GetImageFromArray(output_array_itk), "./output/output_itk.tif")
        ###########################################################################

    @parameterized.expand([TEST_CASE_2D, TEST_CASE_3D])
    def test_arbitary_center_of_rotation(self, filepath):
        # Read image
        image = itk.imread(filepath, itk.F)
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
        output_array_itk = itk_affine_resample(
            image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation
        )

        # MONAI
        metatensor = image_to_metatensor(image)
        affine_matrix_for_monai = itk_to_monai_affine(
            image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation
        )
        output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix_for_monai)

        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor_to_array(metatensor)
        assert np.array_equal(input_array_monai, np.asarray(image))

        ###########################################################################
        # Compare outputs
        print("MONAI-ITK: ", np.allclose(output_array_monai, output_array_itk))

        diff_output = output_array_monai - output_array_itk
        print(f"[Min, Max] MONAI: [{output_array_monai.min()}, {output_array_monai.max()}]")
        print(f"[Min, Max] ITK: [{output_array_itk.min()}, {output_array_itk.max()}]")
        print(f"[Min, Max] diff: [{diff_output.min()}, {diff_output.max()}]")
        ###########################################################################

    @parameterized.expand([TEST_CASE_2D, TEST_CASE_3D])
    def test_monai_to_itk(self, filepath):
        print("\nTEST: MONAI affine matrix -> ITK matrix + translation vector -> transform")
        # Read image
        image = itk.imread(filepath, itk.F)

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
        matrix, translation = monai_to_itk_affine(
            image, affine_matrix=affine_matrix, center_of_rotation=center_of_rotation
        )
        output_array_itk = itk_affine_resample(
            image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation
        )

        # MONAI
        metatensor = image_to_metatensor(image)
        output_array_monai = monai_affine_resample(metatensor, affine_matrix=affine_matrix)

        # Make sure that the array conversion of the inputs is the same
        input_array_monai = metatensor_to_array(metatensor)
        assert np.array_equal(input_array_monai, np.asarray(image))

        ###########################################################################
        # Compare outputs
        print("MONAI-ITK: ", np.allclose(output_array_monai, output_array_itk))

        diff_output = output_array_monai - output_array_itk
        print(f"[Min, Max] MONAI: [{output_array_monai.min()}, {output_array_monai.max()}]")
        print(f"[Min, Max] ITK: [{output_array_itk.min()}, {output_array_itk.max()}]")
        print(f"[Min, Max] diff: [{diff_output.min()}, {diff_output.max()}]")
        ###########################################################################

    @parameterized.expand([TEST_CASE_2D, TEST_CASE_3D])
    def test_cyclic_conversion(self, filepath):
        image = itk.imread(filepath, itk.F)
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

        affine_matrix = itk_to_monai_affine(
            image, matrix=matrix, translation=translation, center_of_rotation=center_of_rotation
        )

        matrix_result, translation_result = monai_to_itk_affine(
            image, affine_matrix=affine_matrix, center_of_rotation=center_of_rotation
        )

        print("Matrix cyclic conversion: ", np.allclose(matrix, matrix_result))
        print("Translation cyclic conversion: ", np.allclose(translation, translation_result))


if __name__ == "__main__":

    # test_utils.download_test_data()

    ## 2D cases
    # filepath0 = str(test_utils.TEST_DATA_DIR / 'CT_2D_head_fixed.mha')
    # filepath1 = str(test_utils.TEST_DATA_DIR / 'CT_2D_head_moving.mha')
    #
    # test_setting_affine_parameters(filepath=filepath0)
    # test_arbitary_center_of_rotation(filepath=filepath0)
    # test_monai_to_itk(filepath=filepath0)
    # test_cyclic_conversion(filepath=filepath0)
    #
    ## 3D cases
    # filepath2 = str(test_utils.TEST_DATA_DIR / 'copd1_highres_INSP_STD_COPD_img.nii.gz')
    # filepath3 = str(test_utils.TEST_DATA_DIR / 'copd1_highres_EXP_STD_COPD_img.nii.gz')
    #
    # test_setting_affine_parameters(filepath=filepath2)
    # test_arbitary_center_of_rotation(filepath=filepath2)
    # test_monai_to_itk(filepath=filepath2)
    # test_cyclic_conversion(filepath=filepath2)

    unittest.main()
