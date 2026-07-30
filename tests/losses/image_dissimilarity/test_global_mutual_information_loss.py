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


class TestGlobalMutualInformationLossHalfPrecision(unittest.TestCase):
    """Test stable Gaussian mutual information in reduced-precision modes."""

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_half_precision_gaussian_weights_with_many_bins_are_finite(self, dtype):
        """Verify many-bin Parzen outputs remain finite and preserve metadata."""
        image = torch.zeros((1, 1, 2), dtype=dtype)
        loss = GlobalMutualInformationLoss(kernel_type="gaussian", num_bins=256)

        weight, probability = loss.parzen_windowing_gaussian(image)

        self.assertTrue(torch.isfinite(weight).all())
        self.assertTrue(torch.isfinite(probability).all())
        self.assertEqual(weight.dtype, image.dtype)
        self.assertEqual(probability.dtype, image.dtype)
        self.assertEqual(weight.device, image.device)
        self.assertEqual(probability.device, image.device)
        torch.testing.assert_close(
            weight.float().sum(dim=-1), torch.ones_like(weight[..., 0], dtype=torch.float32), rtol=0.0, atol=5e-3
        )
        torch.testing.assert_close(
            probability.float().sum(dim=-1),
            torch.ones_like(probability[..., 0], dtype=torch.float32),
            rtol=0.0,
            atol=5e-3,
        )

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_module_cast_with_many_bins_remains_finite(self, dtype):
        """Verify module dtype conversion cannot overflow Gaussian parameters.

        Args:
            dtype: reduced-precision floating-point dtype to test.
        """
        image = torch.linspace(0.0, 1.0, 64, dtype=dtype).reshape(1, 1, 8, 8).requires_grad_()
        target = torch.flip(image.detach(), dims=(-1,))
        loss = GlobalMutualInformationLoss(kernel_type="gaussian", num_bins=256).to(dtype=dtype)

        weight, probability = loss.parzen_windowing_gaussian(image)
        result = loss(image, target)

        self.assertTrue(torch.isfinite(weight).all())
        self.assertTrue(torch.isfinite(probability).all())
        self.assertTrue(torch.isfinite(result))
        result.backward()
        self.assertIsNotNone(image.grad)
        self.assertTrue(torch.isfinite(image.grad).all())

    def test_float16_default_dtype_with_many_bins_remains_finite(self):
        """Verify construction under a float16 default keeps Gaussian parameters finite."""
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float16)
            image = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8).requires_grad_()
            target = torch.flip(image.detach(), dims=(-1,))
            loss = GlobalMutualInformationLoss(kernel_type="gaussian", num_bins=256)

            weight, probability = loss.parzen_windowing_gaussian(image)
            result = loss(image, target)

            self.assertTrue(torch.isfinite(weight).all())
            self.assertTrue(torch.isfinite(probability).all())
            self.assertTrue(torch.isfinite(result))
            result.backward()
            self.assertIsNotNone(image.grad)
            self.assertTrue(torch.isfinite(image.grad).all())
        finally:
            torch.set_default_dtype(original_dtype)

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_half_precision_nonconstant_images_match_float32(self, dtype):
        """Verify nonconstant reduced-precision loss tracks float32.

        Args:
            dtype: reduced-precision floating-point dtype to test.
        """
        pred_float = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8)
        target_float = torch.flip(pred_float, dims=(-1,))
        loss = GlobalMutualInformationLoss(kernel_type="gaussian")
        expected = loss(pred_float, target_float)
        pred = pred_float.to(dtype=dtype).requires_grad_()
        target = target_float.to(dtype=dtype)

        result = loss(pred, target)

        self.assertTrue(torch.isfinite(result))
        self.assertEqual(result.dtype, dtype)
        torch.testing.assert_close(result.float(), expected, rtol=1e-2, atol=1e-2)
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_half_precision_weak_mutual_information_matches_float32(self, dtype):
        """Verify weak reduced-precision mutual information tracks float32.

        Args:
            dtype: reduced-precision floating-point dtype to test.
        """
        generator = torch.Generator().manual_seed(19)
        pred_float = torch.rand((1, 1, 4096), generator=generator)
        target_float = torch.rand((1, 1, 4096), generator=generator)
        loss = GlobalMutualInformationLoss(kernel_type="gaussian", num_bins=8)
        expected = loss(pred_float, target_float)
        pred = pred_float.to(dtype=dtype).requires_grad_()
        target = target_float.to(dtype=dtype)

        result = loss(pred, target)

        self.assertEqual(result.dtype, dtype)
        torch.testing.assert_close(result.float(), expected, rtol=1e-2, atol=1e-6)
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    @parameterized.expand([(torch.float16,), (torch.bfloat16,)])
    def test_half_precision_large_constant_volume_is_finite(self, dtype):
        """Verify reduced-precision loss and gradients remain finite."""
        pred = torch.zeros((1, 1, 48, 48, 48), dtype=dtype, requires_grad=True)
        target = torch.zeros_like(pred)
        loss = GlobalMutualInformationLoss(kernel_type="gaussian")

        result = loss(pred, target)

        self.assertTrue(torch.isfinite(result))
        self.assertEqual(result.dtype, pred.dtype)
        self.assertEqual(result.device, pred.device)
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad.dtype, pred.dtype)
        self.assertEqual(pred.grad.device, pred.device)

    def test_cpu_float16_autocast_nonconstant_images_match_float32(self):
        """Verify nonconstant CPU autocast loss matches float32."""
        pred = torch.linspace(0.0, 1.0, 64).reshape(1, 1, 8, 8).requires_grad_()
        target = torch.flip(pred.detach(), dims=(-1,))
        loss = GlobalMutualInformationLoss(kernel_type="gaussian")
        expected = loss(pred, target).detach()

        with torch.autocast(device_type="cpu", dtype=torch.float16):
            result = loss(pred, target)

        self.assertTrue(torch.isfinite(result))
        self.assertEqual(result.dtype, pred.dtype)
        torch.testing.assert_close(result, expected)
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_scripted_cpu_float16_autocast_large_volume_is_finite(self):
        """Verify scripted loss avoids float16 histogram overflow under autocast."""
        pred = torch.zeros((1, 1, 257, 257), requires_grad=True)
        target = torch.zeros_like(pred)
        loss = torch.jit.script(GlobalMutualInformationLoss(kernel_type="gaussian"))

        with torch.autocast(device_type="cpu", dtype=torch.float16):
            result = loss(pred, target)

        self.assertTrue(torch.isfinite(result))
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_cpu_float16_autocast_large_volume_is_finite(self):
        """Verify CPU float16 autocast avoids histogram accumulation overflow."""
        pred = torch.zeros((1, 1, 48, 48, 48), requires_grad=True)
        target = torch.zeros_like(pred)
        loss = GlobalMutualInformationLoss(kernel_type="gaussian")

        with torch.autocast(device_type="cpu", dtype=torch.float16):
            result = loss(pred, target)

        self.assertTrue(torch.isfinite(result))
        self.assertEqual(result.dtype, pred.dtype)
        result.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad.dtype, pred.dtype)
        self.assertEqual(pred.grad.device, pred.device)


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
