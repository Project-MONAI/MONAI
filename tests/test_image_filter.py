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


import unittest

import torch
import numpy as np
from parameterized import parameterized
from monai.networks.layers.simplelayers import GaussianFilter

from monai.transforms import ImageFilter, ImageFilterd, RandImageFilter, RandImageFilterd

EXPECTED_KERNELS = {
    "mean": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).float(),
    "laplacian": torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).float(),
    "elliptical": torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).float(),
    "sharpen": torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]).float(),
}

SUPPORTED_KERNELS = ["mean", "laplace", "elliptical", "sobel", "sharpen", "median", "gauss", "savitzky_golay"]
SAMPLE_IMAGE_2D = torch.randn(1, 10, 10)
SAMPLE_IMAGE_3D = torch.randn(1, 10, 10, 10)
SAMPLE_DICT = {"image_2d": SAMPLE_IMAGE_2D, "image_3d": SAMPLE_IMAGE_3D}

ADDITIONAL_ARGUMENTS = {
    "order": 1,
    "sigma": 1
    }

class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1

class TestNotAModuleOrTransform:
    pass

class TestImageFilter(unittest.TestCase):
    @parameterized.expand(SUPPORTED_KERNELS)
    def test_init_from_string(self, kernel_name):
        "Test init from string"
        _ = ImageFilter(kernel_name, 3, **ADDITIONAL_ARGUMENTS)

    def test_init_raises(self):
        with self.assertRaises(Exception) as context:
            _ = ImageFilter("mean")
            self.assertTrue(
                "`filter_size` must be specified when specifying filters by string." in str(context.output)
            )
        with self.assertRaises(Exception) as context:
            _ = ImageFilter("mean")
            self.assertTrue(
                "`filter_size` should be a single uneven integer." in str(context.output)
            )
        with self.assertRaises(Exception) as context:
            _ = ImageFilter("gauss", 3)
            self.assertTrue(
                "`filter='gauss', requires the additonal keyword argument `sigma`" in str(context.output)
            )
        with self.assertRaises(Exception) as context:
            _ = ImageFilter("savitzky_golay", 3)
            self.assertTrue(
                "`filter='savitzky_golay', requires the additonal keyword argument `order`" in str(context.output)
            )

    def test_init_from_array(self):
        "Test init with custom kernel and assert wrong kernel shape throws an error"
        _ = ImageFilter(torch.ones(3, 3))
        _ = ImageFilter(torch.ones(3, 3, 3))
        _ = ImageFilter(np.ones((3, 3)))
        _ = ImageFilter(np.ones((3, 3, 3)))

        with self.assertRaises(Exception) as context:
            _ = ImageFilter(torch.ones(3, 3, 3, 3))
            self.assertTrue(
                "Only 1D, 2D, and 3D filters are supported." in str(context.output)
            )

    def test_init_from_module(self):
        filter = ImageFilter(TestModule())
        out = filter(torch.zeros(1,3,3,3))
        torch.testing.assert_allclose(torch.ones(1,3,3,3), out)

    def test_init_from_transform(self):
        _ = ImageFilter(GaussianFilter(3, sigma = 2))

    def test_init_from_wrong_type_fails(self):
        with self.assertRaises(Exception) as context:
            _ = ImageFilter(TestNotAModuleOrTransform())
            self.assertTrue(
                "<class 'type'> is not supported." in str(context.output)
            )

    @parameterized.expand(EXPECTED_KERNELS.keys())
    def test_2d_kernel_correctness(self, kernel_name):
        "Test correctness of kernels (2d only)"
        tfm = ImageFilter(kernel_name, kernel_size=3)
        kernel = tfm._create_kernel_from_string(kernel_name, size=3, ndim=2).squeeze()
        torch.testing.assert_allclose(kernel, EXPECTED_KERNELS[kernel_name])

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d(self, kernel_name):
        "Text function `__call__` for 2d images"
        filter = ImageFilter(kernel_name, 3)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_IMAGE_2D)
            self.assertEqual(out_tensor.shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d(self, kernel_name):
        "Text function `__call__` for 3d images"
        filter = ImageFilter(kernel_name, 3)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        self.assertEqual(out_tensor.shape, SAMPLE_IMAGE_3D.shape)


class TestImageFilterDict(unittest.TestCase):
    @parameterized.expand(SUPPORTED_KERNELS)
    def test_init_from_string_dict(self, kernel_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = ImageFilterd("image", kernel_name, 3)
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = ImageFilterd(self.image_key, kernel_name)

    def test_init_from_array_dict(self):
        "Test init with custom kernel and assert wrong kernel shape throws an error"
        _ = ImageFilterd("image", torch.ones(3, 3))
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = ImageFilterd(self.image_key, torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d(self, kernel_name):
        "Text function `__call__` for 2d images"
        filter = ImageFilterd("image_2d", kernel_name, 3)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_DICT)
            self.assertEqual(out_tensor["image_2d"].shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d(self, kernel_name):
        "Text function `__call__` for 3d images"
        filter = ImageFilterd("image_3d", kernel_name, 3)
        out_tensor = filter(SAMPLE_DICT)
        self.assertEqual(out_tensor["image_3d"].shape, SAMPLE_IMAGE_3D.shape)


class TestRandImageFilter(unittest.TestCase):
    @parameterized.expand(SUPPORTED_KERNELS)
    def test_init_from_string(self, kernel_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = RandImageFilter(kernel_name, 3)
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = RandImageFilter(kernel_name)

    def test_init_from_array(self):
        "Test init with custom kernel and assert wrong kernel shape throws an error"
        _ = RandImageFilter(torch.ones(3, 3))
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = RandImageFilter(torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d_prob_1(self, kernel_name):
        "Text function `__call__` for 2d images"
        filter = RandImageFilter(kernel_name, 3, 1)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_IMAGE_2D)
            self.assertEqual(out_tensor.shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d_prob_1(self, kernel_name):
        "Text function `__call__` for 3d images"
        filter = RandImageFilter(kernel_name, 3, 1)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        self.assertEqual(out_tensor.shape, SAMPLE_IMAGE_3D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d_prob_0(self, kernel_name):
        "Text function `__call__` for 2d images"
        filter = RandImageFilter(kernel_name, 3, 0)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_IMAGE_2D)
            torch.testing.assert_allclose(out_tensor, SAMPLE_IMAGE_2D)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d_prob_0(self, kernel_name):
        "Text function `__call__` for 3d images"
        filter = RandImageFilter(kernel_name, 3, 0)
        out_tensor = filter(SAMPLE_IMAGE_3D)
        torch.testing.assert_allclose(out_tensor, SAMPLE_IMAGE_3D)


class TestRandImageFilterDict(unittest.TestCase):
    @parameterized.expand(SUPPORTED_KERNELS)
    def test_init_from_string_dict(self, kernel_name):
        "Test init from string and assert an error is thrown if no size is passed"
        _ = RandImageFilterd("image", kernel_name, 3)
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = RandImageFilterd("image", kernel_name)

    def test_init_from_array_dict(self):
        "Test init with custom kernel and assert wrong kernel shape throws an error"
        _ = RandImageFilterd("image", torch.ones(3, 3))
        with self.assertRaises(Exception) as context:  # noqa F841
            _ = RandImageFilterd("image", torch.ones(3, 3, 3, 3))

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d_prob_1(self, kernel_name):
        filter = RandImageFilterd("image_2d", kernel_name, 3, 1.0)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_DICT)
            self.assertEqual(out_tensor["image_2d"].shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d_prob_1(self, kernel_name):
        filter = RandImageFilterd("image_3d", kernel_name, 3, 1.0)
        out_tensor = filter(SAMPLE_DICT)
        self.assertEqual(out_tensor["image_3d"].shape, SAMPLE_IMAGE_3D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_2d_prob_0(self, kernel_name):
        filter = RandImageFilterd("image_2d", kernel_name, 3, 0.0)
        if kernel_name != "sobel_d":  # sobel_d does not support 2d
            out_tensor = filter(SAMPLE_DICT)
            torch.testing.assert_allclose(out_tensor["image_2d"].shape, SAMPLE_IMAGE_2D.shape)

    @parameterized.expand(SUPPORTED_KERNELS)
    def test_call_3d_prob_0(self, kernel_name):
        filter = RandImageFilterd("image_3d", kernel_name, 3, 0.0)
        out_tensor = filter(SAMPLE_DICT)
        torch.testing.assert_allclose(out_tensor["image_3d"].shape, SAMPLE_IMAGE_3D.shape)
