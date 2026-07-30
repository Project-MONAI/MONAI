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
from pathlib import Path

import numpy as np
import torch
from parameterized import parameterized

from monai import transforms
from monai.losses.image_dissimilarity import GlobalMutualInformationLoss
from tests.test_utils import download_url_or_skip_test, skip_if_quick, testing_data_config

device = "cuda:0" if torch.cuda.is_available() else "cpu"

TESTS_PATH = Path(__file__).parents[2]
FILE_PATH = os.path.join(TESTS_PATH, "testing_data", "temp_" + "mri.nii")

EXPECTED_VALUE = {
    "xyz_translation": [
        -1.5860257,
        -0.62433463,
        -0.38217825,
        -0.2905613,
        -0.23233329,
        -0.1961407,
        -0.16905619,
        -0.15100679,
        -0.13666219,
        -0.12635908,
    ],
    "xyz_rotation": [
        -1.5860257,
        -0.30265224,
        -0.18666176,
        -0.15887907,
        -0.1625064,
        -0.16603896,
        -0.19222091,
        -0.18158069,
        -0.167644,
        -0.16698098,
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
                transforms.LoadImaged(keys="img", image_only=True),
                transforms.Affined(
                    keys="img",
                    translate_params=translate_params,
                    rotate_params=rotate_params,
                    device=None,
                    padding_mode="border",
                ),
                transforms.NormalizeIntensityd(keys=["img"]),
            ]
            transformation = transforms.Compose(transform_list)
            return transformation({"img": FILE_PATH})["img"]

        a1 = transformation()
        a1 = a1.clone().unsqueeze(0).unsqueeze(0).to(device)

        for mode in transform_params_dict:
            transform_params_list = transform_params_dict[mode]
            expected_value_list = EXPECTED_VALUE[mode]
            for transform_params, expected_value in zip(transform_params_list, expected_value_list):
                a2 = transformation(
                    translate_params=transform_params if "translation" in mode else (0.0, 0.0, 0.0),
                    rotate_params=transform_params if "rotation" in mode else (0.0, 0.0, 0.0),
                )
                a2 = a2.clone().unsqueeze(0).unsqueeze(0).to(device)
                result = loss_fn(a2, a1).detach().cpu().numpy()
                np.testing.assert_allclose(result, expected_value, rtol=0.08, atol=0.05)


class TestGlobalMutualInformationLossIll(unittest.TestCase):
    def test_gaussian_bin_centers_registered_buffer(self):
        loss = GlobalMutualInformationLoss(kernel_type="gaussian", num_bins=16)

        self.assertIn("bin_centers", dict(loss.named_buffers()))
        self.assertIsNotNone(loss.bin_centers)
        self.assertFalse(loss.bin_centers.requires_grad)

        loss = loss.to(dtype=torch.float64)
        self.assertEqual(loss.bin_centers.dtype, torch.float64)

        if torch.cuda.is_available():
            loss = loss.to(device="cuda:0")
            self.assertEqual(loss.bin_centers.device, torch.device("cuda:0"))

    def test_b_spline_bin_centers_exists_as_none(self):
        loss = GlobalMutualInformationLoss(kernel_type="b-spline")

        self.assertIsNone(loss.bin_centers)

    def test_b_spline_num_bins_must_allow_padding(self):
        """Verify B-spline bin counts leave room for boundary padding."""
        with self.assertRaisesRegex(ValueError, "num_bins must be greater than 4"):
            GlobalMutualInformationLoss(kernel_type="b-spline", num_bins=4)

    @parameterized.expand(
        [
            (torch.ones((1, 2), dtype=torch.float), torch.ones((1, 3), dtype=torch.float)),  # mismatched_simple_dims
            (
                torch.ones((1, 3, 3), dtype=torch.float),
                torch.ones((1, 3), dtype=torch.float),
            ),  # mismatched_advanced_dims
        ]
    )
    def test_ill_shape(self, input1, input2):
        loss = GlobalMutualInformationLoss()
        with self.assertRaises(ValueError):
            loss.forward(input1, input2)

    @parameterized.expand(
        [
            (0, "mean", ValueError, ""),  # num_bins_zero
            (-1, "mean", ValueError, ""),  # num_bins_negative
            (64, "unknown", ValueError, ""),  # reduction_unknown
            (64, None, ValueError, ""),  # reduction_none
        ]
    )
    def test_ill_opts(self, num_bins, reduction, expected_exception, expected_message):
        pred = torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device)
        target = torch.ones((1, 3, 3, 3, 3), dtype=torch.float, device=device)
        with self.assertRaisesRegex(expected_exception, expected_message):
            GlobalMutualInformationLoss(num_bins=num_bins, reduction=reduction)(pred, target)


class TestGlobalMutualInformationLossBSpline(unittest.TestCase):
    """Test B-spline mutual information on degenerate intensity ranges."""

    @parameterized.expand(["prediction", "target"])
    def test_b_spline_single_constant_input_is_finite(self, constant_input):
        """Verify either independently constant input yields finite gradients.

        Args:
            constant_input: Which input (``"prediction"`` or ``"target"``)
                is held constant.
        """
        varying = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)
        if constant_input == "prediction":
            pred = torch.zeros_like(varying, requires_grad=True)
            target = varying
        else:
            pred = varying.clone().requires_grad_()
            target = torch.ones_like(varying)
        loss = GlobalMutualInformationLoss(kernel_type="b-spline")

        result = loss(pred, target)

        self.assertTrue(torch.isfinite(result))
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_b_spline_constant_half_precision_images_are_finite(self):
        """Verify constant float16 inputs survive overflow-sized bin distances."""
        pred = torch.zeros((1, 1, 8, 8), dtype=torch.float16, requires_grad=True)
        target = torch.ones_like(pred)
        loss = GlobalMutualInformationLoss(kernel_type="b-spline", num_bins=64)

        result = loss(pred, target)

        self.assertTrue(torch.isfinite(result))
        self.assertAlmostEqual(result.item(), 0.0, places=6)
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_b_spline_float16_small_range_preserves_signal(self):
        """Verify a small nonzero float16 range retains loss and gradient signal."""
        values = torch.linspace(0.0, 5e-4, 64, dtype=torch.float16).reshape(1, 1, 8, 8)
        pred = values.clone().requires_grad_()
        loss = GlobalMutualInformationLoss(kernel_type="b-spline", num_bins=64)

        result = loss(pred, values)

        self.assertTrue(torch.isfinite(result))
        self.assertLess(result.item(), -1.0)
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(torch.count_nonzero(pred.grad).item(), 0)

    @parameterized.expand(
        [
            ("float16_tiny", torch.float16, torch.finfo(torch.float16).tiny / 2),
            ("bfloat16_tiny", torch.bfloat16, torch.finfo(torch.bfloat16).tiny / 2),
            ("float32_tiny", torch.float32, torch.finfo(torch.float32).tiny / 2),
            ("float16_large", torch.float16, 65000.0),
        ]
    )
    def test_b_spline_nonzero_ranges_are_finite(self, case_name, dtype, maximum):
        """Verify extreme nonzero ranges yield finite loss and gradients.

        Args:
            case_name: Descriptive label for the parameterized range case.
            dtype: Tensor dtype used for the prediction and target.
            maximum: Nonzero upper endpoint of the tested intensity range.
        """
        values = torch.tensor([0.0, maximum, maximum, 0.0], dtype=dtype).reshape(1, 1, 2, 2)
        pred = values.clone().requires_grad_()
        target = torch.flip(values, dims=(-1,))
        loss = GlobalMutualInformationLoss(kernel_type="b-spline", num_bins=64)

        result = loss(pred, target)

        self.assertTrue(torch.isfinite(result))
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())


class TestGlobalMutualInformationLossBuffers(unittest.TestCase):
    def test_gaussian_kernel_registers_buffers(self):
        """Verify gaussian kernel registers preterm and bin_centers as non-trainable, non-persistent buffers."""
        loss = GlobalMutualInformationLoss(kernel_type="gaussian")
        self.assertIn("preterm", loss._buffers)
        self.assertIn("bin_centers", loss._buffers)
        self.assertFalse(loss.preterm.requires_grad)
        self.assertFalse(loss.bin_centers.requires_grad)
        self.assertEqual(loss.bin_centers.ndim, 3)
        state = loss.state_dict()
        self.assertNotIn("preterm", state)
        self.assertNotIn("bin_centers", state)

    def test_bspline_kernel_has_no_gaussian_buffers(self):
        """Verify b-spline kernel does not populate gaussian-specific buffers."""
        loss = GlobalMutualInformationLoss(kernel_type="b-spline")
        self.assertIsNone(loss.preterm)
        self.assertIsNone(loss.bin_centers)
        state = loss.state_dict()
        self.assertNotIn("preterm", state)
        self.assertNotIn("bin_centers", state)

    def test_gaussian_kernel_forward_correct(self):
        """Verify gaussian kernel forward pass returns a scalar loss tensor."""
        pred = torch.rand(2, 1, 8, 8, dtype=torch.float32)
        target = torch.rand(2, 1, 8, 8, dtype=torch.float32)
        loss = GlobalMutualInformationLoss(kernel_type="gaussian")
        result = loss(pred, target)
        self.assertEqual(result.shape, torch.Size([]))

    def test_gaussian_buffers_move_with_module(self):
        """Verify preterm and bin_centers buffers move to the target device with the module."""
        loss = GlobalMutualInformationLoss(kernel_type="gaussian")
        self.assertEqual(loss.preterm.device.type, "cpu")
        self.assertEqual(loss.bin_centers.device.type, "cpu")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        loss = loss.cuda()
        self.assertEqual(loss.preterm.device.type, "cuda")
        self.assertEqual(loss.bin_centers.device.type, "cuda")
        pred = torch.rand(2, 1, 8, 8, device="cuda")
        target = torch.rand(2, 1, 8, 8, device="cuda")
        result = loss(pred, target)
        self.assertEqual(result.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
