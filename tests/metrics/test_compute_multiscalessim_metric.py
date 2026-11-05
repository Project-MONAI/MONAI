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

import unittest

import torch
from parameterized import parameterized

from monai.metrics import MultiScaleSSIMMetric
from monai.utils import set_determinism


class TestMultiScaleSSIMMetric(unittest.TestCase):
    def test_four_scales_single_pixel_kernel(self):
        """Do not downsample after the final scale reaches one pixel."""
        image = torch.ones(1, 1, 8, 8)
        metric = MultiScaleSSIMMetric(spatial_dims=2, kernel_size=1, weights=[0.25] * 4)

        torch.testing.assert_close(metric(image, image), torch.ones(1, 1))

    def test_2d_gaussian(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(1, 1, 64, 64))
        target = torch.abs(torch.randn(1, 1, 64, 64))
        preds = preds / preds.max()
        target = target / target.max()

        metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_type="gaussian", weights=[0.5, 0.5])
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.023176
        self.assertAlmostEqual(expected_value, result.item(), 4)

    def test_2d_uniform(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(1, 1, 64, 64))
        target = torch.abs(torch.randn(1, 1, 64, 64))
        preds = preds / preds.max()
        target = target / target.max()

        metric = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_type="uniform", weights=[0.5, 0.5])
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.022655
        self.assertAlmostEqual(expected_value, result.item(), 4)

    def test_3d_gaussian(self):
        set_determinism(0)
        preds = torch.abs(torch.randn(1, 1, 64, 64, 64))
        target = torch.abs(torch.randn(1, 1, 64, 64, 64))
        preds = preds / preds.max()
        target = target / target.max()

        metric = MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_type="gaussian", weights=[0.5, 0.5])
        metric(preds, target)
        result = metric.aggregate()
        expected_value = 0.061796
        self.assertAlmostEqual(expected_value, result.item(), 4)

    def test_input_ill_input_shape2d(self):
        metric = MultiScaleSSIMMetric(spatial_dims=3, weights=[0.5, 0.5])

        with self.assertRaises(ValueError):
            metric(torch.randn(1, 1, 64, 64), torch.randn(1, 1, 64, 64))

    def test_input_ill_input_shape3d(self):
        metric = MultiScaleSSIMMetric(spatial_dims=2, weights=[0.5, 0.5])

        with self.assertRaises(ValueError):
            metric(torch.randn(1, 1, 64, 64, 64), torch.randn(1, 1, 64, 64, 64))

    @parameterized.expand([(spatial_dims, scales) for spatial_dims in (2, 3) for scales in range(1, 7)])
    def test_minimum_input_size(self, spatial_dims, scales):
        kernel_size = 3
        minimum_size = kernel_size * 2 ** (scales - 1)
        image = torch.ones(1, 1, *([minimum_size] * spatial_dims))
        metric = MultiScaleSSIMMetric(
            spatial_dims=spatial_dims, kernel_size=kernel_size, weights=[1.0 / scales] * scales
        )

        self.assertAlmostEqual(metric(image, image).item(), 1.0)

    @parameterized.expand([(spatial_dims, scales) for spatial_dims in (2, 3) for scales in range(1, 7)])
    def test_undersized_spatial_dimension(self, spatial_dims, scales):
        kernel_size = (3, 5) if spatial_dims == 2 else (3, 5, 3)
        minimum_shape = [size * 2 ** (scales - 1) for size in kernel_size]
        metric = MultiScaleSSIMMetric(
            spatial_dims=spatial_dims, kernel_size=kernel_size, weights=[1.0 / scales] * scales
        )

        for axis in range(spatial_dims):
            with self.subTest(axis=axis):
                shape = minimum_shape.copy()
                shape[axis] -= 1
                image = torch.ones(1, 1, *shape)
                message = f"spatial dimension {axis} of the image must be at least {minimum_shape[axis]}"
                with self.assertRaisesRegex(ValueError, message):
                    metric(image, image)

    def test_four_scales_default_kernel(self):
        image = torch.ones(1, 1, 88, 88)
        metric = MultiScaleSSIMMetric(spatial_dims=2, weights=[0.25] * 4)

        self.assertAlmostEqual(metric(image, image).item(), 1.0)

    def test_two_scales_too_small(self):
        image = torch.ones(1, 1, 16, 16)
        metric = MultiScaleSSIMMetric(spatial_dims=2, weights=[0.5, 0.5])

        with self.assertRaisesRegex(ValueError, "spatial dimension 0 of the image must be at least 22"):
            metric(image, image)

    def test_small_inputs(self):
        metric = MultiScaleSSIMMetric(spatial_dims=2)

        with self.assertRaises(ValueError):
            metric(torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 16, 16, 16))


if __name__ == "__main__":
    unittest.main()
