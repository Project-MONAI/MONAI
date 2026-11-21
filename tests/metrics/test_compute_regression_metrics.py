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
from functools import partial

import numpy as np
import torch
from parameterized import parameterized

from monai.metrics import MAEMetric, MAPEMetric, MSEMetric, MultiScaleSSIMMetric, PSNRMetric, RMSEMetric, SSIMMetric
from monai.utils import set_determinism


# define a numpy flatten function that only preserves batch dimension
def flatten(data):
    return np.reshape(data, [data.shape[0], -1])


# define metrics computation truth functions to check our monai metrics against
def msemetric_np(y_pred, y):
    return np.mean((flatten(y_pred) - flatten(y)) ** 2)


def maemetric_np(y_pred, y):
    return np.mean(np.abs(flatten(y_pred) - flatten(y)))


def rmsemetric_np(y_pred, y):
    return np.mean(np.sqrt(np.mean((flatten(y_pred) - flatten(y)) ** 2, axis=1)))


def psnrmetric_np(max_val, y_pred, y):
    mse = np.mean((flatten(y_pred) - flatten(y)) ** 2, axis=1)
    return np.mean(20 * np.log10(max_val) - 10 * np.log10(mse))


def mapemetric_np(y_pred, y, epsilon=1e-7):
    percentage_error = np.abs(y - y_pred) / np.clip(np.abs(y), a_min=epsilon, a_max=None) * 100.0
    return np.mean(flatten(percentage_error))


class TestRegressionMetrics(unittest.TestCase):

    def test_shape_reduction(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check
        metrics = [MSEMetric, MAEMetric, MAPEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for batch in batch_dims:
            for spatial in spatial_dims:
                for base in base_dims:
                    # create random tensors
                    in_tensor = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)

                    # iterate over regression metrics, check shape for diff. reduction func
                    for mt_fn in metrics:
                        mt = mt_fn(reduction="mean")
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate()
                        self.assertEqual(len(out_tensor.shape), 1)

                        mt = mt_fn(reduction="sum")
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate()
                        self.assertEqual(len(out_tensor.shape), 0)

                        mt = mt_fn(reduction="sum")  # test reduction arg overriding
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate(reduction="mean_channel")
                        self.assertEqual(len(out_tensor.shape), 1)
                        self.assertEqual(out_tensor.shape[0], batch)

                        mt = mt_fn(reduction="sum_channel")
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate()
                        self.assertEqual(len(out_tensor.shape), 1)
                        self.assertEqual(out_tensor.shape[0], batch)

    def test_compare_numpy(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check + truth metric function in numpy
        metrics = [MSEMetric, MAEMetric, MAPEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        metrics_np = [msemetric_np, maemetric_np, mapemetric_np, rmsemetric_np, partial(psnrmetric_np, max_val=1.0)]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for batch in batch_dims:
            for spatial in spatial_dims:
                for base in base_dims:
                    # create random tensors
                    in_tensor_a = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)
                    in_tensor_b = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)

                    # check metrics
                    for mt_fn, mt_fn_np in zip(metrics, metrics_np):
                        mt = mt_fn()
                        mt(y_pred=in_tensor_a, y=in_tensor_b)
                        out_tensor = mt.aggregate(reduction="mean")
                        out_np = mt_fn_np(y_pred=in_tensor_a.cpu().numpy(), y=in_tensor_b.cpu().numpy())

                        np.testing.assert_allclose(out_tensor.cpu().numpy(), out_np, atol=1e-3, rtol=1e-4)

    def test_ill_shape(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # regression metrics to check + truth metric function in numpy
        metrics = [MSEMetric, MAEMetric, MAPEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        basedim = 10

        # too small shape
        with self.assertRaises(ValueError):
            in_tensor = torch.rand((basedim,)).to(device)
            for mt_fn in metrics:
                mt_fn()(in_tensor, in_tensor)

        # different shape for pred/target
        with self.assertRaises(ValueError):
            in_tensor_a = torch.rand((basedim,)).to(device)
            in_tensor_b = torch.rand((basedim, basedim)).to(device)
            for mt_fn in metrics:
                mt_fn()(y_pred=in_tensor_a, y=in_tensor_b)

    def test_same_input(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = [MSEMetric, MAEMetric, MAPEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        results = [0.0, 0.0, 0.0, 0.0, float("inf")]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for batch in batch_dims:
            for spatial in spatial_dims:
                for base in base_dims:
                    # create random tensors
                    in_tensor = torch.rand((batch,) + (base,) * (spatial - 1)).to(device)

                    # check metrics
                    for mt_fn, rs in zip(metrics, results):
                        mt = mt_fn(reduction="mean")
                        mt(in_tensor, in_tensor)
                        out_tensor = mt.aggregate()
                        np.testing.assert_allclose(out_tensor.cpu(), rs, atol=1e-4)

    def test_diff_input(self):
        set_determinism(seed=123)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metrics = [MSEMetric, MAEMetric, MAPEMetric, RMSEMetric, partial(PSNRMetric, max_val=1.0)]
        results = [1.0, 1.0, 100.0, 1.0, 0.0]

        # define variations in batch/base_dims/spatial_dims
        batch_dims = [1, 2, 4, 16]
        base_dims = [16, 32, 64]
        spatial_dims = [2, 3, 4]

        # iterate over all variations and check shapes for different reduction functions
        for batch in batch_dims:
            for spatial in spatial_dims:
                for base in base_dims:
                    # create random tensors
                    in_tensor_a = torch.zeros((batch,) + (base,) * (spatial - 1)).to(device)
                    in_tensor_b = torch.ones((batch,) + (base,) * (spatial - 1)).to(device)

                    # check metrics
                    for mt_fn, rs in zip(metrics, results):
                        mt = mt_fn(reduction="mean")
                        mt(in_tensor_a, in_tensor_b)
                        out_tensor = mt.aggregate()
                        np.testing.assert_allclose(out_tensor.cpu(), rs, atol=1e-4)


class TestRegressionMetricsIntegerInput(unittest.TestCase):
    """Check small-integer calculations without reducing existing floating-point precision."""

    @parameterized.expand(
        [
            (f"{metric}_{name}_{'list' if as_list else 'batch'}", metric, dtype, lower, upper, as_list)
            for metric in ("mse", "mae", "mape", "rmse", "psnr")
            for name, dtype, lower, upper in (
                ("uint8", torch.uint8, 0, 255),
                ("int8", torch.int8, -128, 127),
                ("int16", torch.int16, -32768, 32767),
            )
            for as_list in (False, True)
        ]
    )
    def test_small_integer_input(self, _, metric_name, dtype, lower, upper, as_list):
        """Match hand-computed errors for both subtraction directions and public input forms.

        Args:
            _: Parameterized case name.
            metric_name: Regression metric to evaluate.
            dtype: Small integer dtype shared by the two inputs.
            lower: Minimum input value.
            upper: Maximum input value.
            as_list: Whether to pass a list of channel-first images instead of a batch.
        """
        prediction = torch.tensor([upper, lower], dtype=dtype).reshape(2, 1, 1, 1).expand(2, 1, 2, 3)
        target = torch.tensor([lower, upper], dtype=dtype).reshape(2, 1, 1, 1).expand(2, 1, 2, 3)
        difference = upper - lower
        metric_types = {"mse": MSEMetric, "mae": MAEMetric, "mape": MAPEMetric, "rmse": RMSEMetric, "psnr": PSNRMetric}
        kwargs = {"max_val": difference} if metric_name == "psnr" else {"epsilon": 1.0} if metric_name == "mape" else {}
        expected = {
            "mse": [[difference**2], [difference**2]],
            "mae": [[difference], [difference]],
            "mape": [[100.0 * difference / max(abs(lower), 1)], [100.0 * difference / max(abs(upper), 1)]],
            "rmse": [[difference], [difference]],
            "psnr": [[0.0], [0.0]],
        }[metric_name]
        metric = metric_types[metric_name](**kwargs)
        result = metric(list(prediction) if as_list else prediction, list(target) if as_list else target)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-4)
        np.testing.assert_allclose(metric.aggregate().item(), np.mean(expected), rtol=1e-6, atol=1e-4)
        self.assertEqual(prediction.dtype, dtype)
        self.assertEqual(target.dtype, dtype)

    @parameterized.expand(
        [
            (f"{metric}_{name}", metric, dtype)
            for metric in ("mse", "mae", "mape", "rmse", "psnr")
            for name, dtype in (("float64_target", torch.float64), ("uint8_target", torch.uint8))
        ]
    )
    def test_float64_precision(self, _, metric_name, target_dtype):
        """Preserve a float64 residual that would disappear after a float32 cast.

        Args:
            _: Parameterized case name.
            metric_name: Regression metric to evaluate.
            target_dtype: Float64 or uint8 dtype of the unit target.
        """
        prediction = torch.tensor([[1.0 + 1e-10]], dtype=torch.float64)
        target = torch.ones((1, 1), dtype=target_dtype)
        difference = prediction.item() - 1.0
        metric_types = {"mse": MSEMetric, "mae": MAEMetric, "mape": MAPEMetric, "rmse": RMSEMetric, "psnr": PSNRMetric}
        metric = metric_types[metric_name](**({"max_val": 1.0} if metric_name == "psnr" else {}))
        expected = {
            "mse": difference**2,
            "mae": difference,
            "mape": 100 * difference,
            "rmse": difference,
            "psnr": -10 * np.log10(difference**2),
        }[metric_name]
        result = metric(prediction, target)
        self.assertEqual(result.dtype, torch.float64)
        np.testing.assert_allclose(result.item(), expected, rtol=1e-12, atol=0)

    def test_float32_precision(self):
        """Keep the calculation dtype of existing float32 inputs."""
        result = MSEMetric()(torch.tensor([[3.0]], dtype=torch.float32), torch.tensor([[1.0]], dtype=torch.float32))
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.item(), 4.0)

    @parameterized.expand([("int32", torch.int32, 2**24), ("int64", torch.int64, 2**53)])
    def test_wider_integer_residual(self, _, dtype, offset):
        """Retain the existing exact subtraction for nearby large integer values.

        Args:
            _: Parameterized case name.
            dtype: Integer dtype outside the small-integer conversion scope.
            offset: Large shared offset that must not hide a unit difference.
        """
        prediction = torch.tensor([[offset + 1]], dtype=dtype)
        target = torch.tensor([[offset]], dtype=dtype)
        self.assertEqual(MSEMetric()(prediction, target).item(), 1.0)

    def test_default_float64(self):
        """Use the default float64 dtype when promoting small integer inputs."""
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            prediction = torch.tensor([[0]], dtype=torch.uint8)
            target = torch.tensor([[255]], dtype=torch.uint8)
            result = MSEMetric()(prediction, target)
            self.assertEqual(result.dtype, torch.float64)
            self.assertEqual(result.item(), 65025.0)
        finally:
            torch.set_default_dtype(original_dtype)

    @parameterized.expand(
        [("ssim_2d", False, 2), ("ssim_3d", False, 3), ("ms_ssim_2d", True, 2), ("ms_ssim_3d", True, 3)]
    )
    def test_integer_image_similarity(self, _, multiscale, spatial_dims):
        """Match equivalent floating images through the shared regression metric entry point.

        Args:
            _: Parameterized case name.
            multiscale: Whether to evaluate MS-SSIM instead of SSIM.
            spatial_dims: Number of image spatial dimensions.
        """
        prediction = torch.arange(16**spatial_dims).remainder(256).to(torch.uint8).reshape(1, 1, *([16] * spatial_dims))
        target = prediction.flip(-1)
        metric_type = MultiScaleSSIMMetric if multiscale else SSIMMetric
        kwargs = {"kernel_size": 3, "weights": (0.5, 0.5)} if multiscale else {"win_size": 3}
        metric = metric_type(spatial_dims=spatial_dims, data_range=255, **kwargs)
        expected = metric_type(spatial_dims=spatial_dims, data_range=255, **kwargs)(prediction.float(), target.float())
        torch.testing.assert_close(metric(prediction, target), expected)


if __name__ == "__main__":
    unittest.main()
