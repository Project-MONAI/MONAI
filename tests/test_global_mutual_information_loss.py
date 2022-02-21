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
import unittest

import numpy as np
import torch

from monai import transforms
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss
from tests.utils import SkipIfBeforePyTorchVersion, download_url_or_skip_test, skip_if_quick, testing_data_config

device = "cuda" if torch.cuda.is_available() else "cpu"

FILE_PATH = os.path.join(os.path.dirname(__file__), "testing_data", "temp_" + "mri.nii")

EXPECTED_VALUE = {
    "xyz_translation": [
        -1.5860259532928467,
        -0.5957175493240356,
        -0.3855515122413635,
        -0.28728482127189636,
        -0.23416118323802948,
        -0.19534644484519958,
        -0.17001715302467346,
        -0.15043553709983826,
        -0.1366637945175171,
        -0.12534910440444946,
    ],
    "xyz_rotation": [
        -1.5860259532928467,
        -0.29977330565452576,
        -0.18411292135715485,
        -0.1582011878490448,
        -0.16107326745986938,
        -0.165723517537117,
        -0.1970357596874237,
        -0.1755618453025818,
        -0.17100191116333008,
        -0.17264796793460846,
    ],
}


@skip_if_quick
class TestGlobalMutualInformationLoss(unittest.TestCase):
    def setUp(self):
        config = testing_data_config("images", "Prostate_T2W_AX_1")
        download_url_or_skip_test(
            url=config["url"],
            filepath=FILE_PATH,
            hash_val=config.get("hash_val"),
            hash_type=config.get("hash_type", "sha256"),
        )

    @SkipIfBeforePyTorchVersion((1, 9))
    def test_bspline(self):
        loss_fn = GlobalMutualInformationLoss(kernel_type="b-spline", num_bins=32, sigma_ratio=0.015)

        transform_params_dict = {
            "xyz_translation": [(i, i, i) for i in range(10)],
            "xyz_rotation": [(np.pi / 100 * i, np.pi / 100 * i, np.pi / 100 * i) for i in range(10)],
        }

        def transformation(translate_params=(0.0, 0.0, 0.0), rotate_params=(0.0, 0.0, 0.0)):
            """
            Read and transform Prostate_T2W_AX_1.nii
            Args:
                translate_params: a tuple of 3 floats, translation is in pixel/voxel relative to the center of the input
                        image. Defaults to no translation.
                rotate_params: a rotation angle in radians, a tuple of 3 floats for 3D.
                        Defaults to no rotation.
            Returns:
                numpy array of shape HWD
            """
            transform_list = [
                transforms.LoadImaged(keys="img"),
                transforms.Affined(
                    keys="img", translate_params=translate_params, rotate_params=rotate_params, device=None
                ),
                transforms.NormalizeIntensityd(keys=["img"]),
            ]
            transformation = transforms.Compose(transform_list)
            return transformation({"img": FILE_PATH})["img"]

        a1 = transformation()
        a1 = torch.tensor(a1).unsqueeze(0).unsqueeze(0).to(device)

        for mode in transform_params_dict.keys():
            transform_params_list = transform_params_dict[mode]
            expected_value_list = EXPECTED_VALUE[mode]
            for transform_params, expected_value in zip(transform_params_list, expected_value_list):
                a2 = transformation(
                    translate_params=transform_params if "translation" in mode else (0.0, 0.0, 0.0),
                    rotate_params=transform_params if "rotation" in mode else (0.0, 0.0, 0.0),
                )
                a2 = torch.tensor(a2).unsqueeze(0).unsqueeze(0).to(device)
                result = loss_fn(a2, a1).detach().cpu().numpy()
                np.testing.assert_allclose(result, expected_value, rtol=1e-3, atol=5e-3)


class TestGlobalMutualInformationLossIll(unittest.TestCase):
    def test_ill_shape(self):
        loss = GlobalMutualInformationLoss()
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 2), dtype=torch.float), torch.ones((1, 3), dtype=torch.float, device=device))
        with self.assertRaisesRegex(ValueError, ""):
            loss.forward(torch.ones((1, 3, 3), dtype=torch.float), torch.ones((1, 3), dtype=torch.float, device=device))

    def test_ill_opts(self):
        pred = torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device)
        target = torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device)
        with self.assertRaisesRegex(ValueError, ""):
            GlobalMutualInformationLoss(num_bins=0)(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            GlobalMutualInformationLoss(num_bins=-1)(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            GlobalMutualInformationLoss(reduction="unknown")(pred, target)
        with self.assertRaisesRegex(ValueError, ""):
            GlobalMutualInformationLoss(reduction=None)(pred, target)


if __name__ == "__main__":
    unittest.main()
