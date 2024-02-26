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

import math
from abc import abstractmethod
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F

from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction, StrEnum, convert_data_type, ensure_tuple_rep
from monai.utils.type_conversion import convert_to_dst_type

from .metric import CumulativeIterationMetric


class RegressionMetric(CumulativeIterationMetric):
    """
    Base class for regression metrics.
    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.
        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f

    def _check_shape(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        if y_pred.shape != y.shape:
            raise ValueError(f"y_pred and y shapes dont match, received y_pred: [{y_pred.shape}] and y: [{y.shape}]")

        # also check if there is atleast one non-batch dimension i.e. num_dims >= 2
        if len(y_pred.shape) < 2:
            raise ValueError("either channel or spatial dimensions required, found only batch dimension")

    @abstractmethod
    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise ValueError("y_pred and y must be PyTorch Tensor.")
        self._check_shape(y_pred, y)
        return self._compute_metric(y_pred, y)


class MSEMetric(RegressionMetric):
    r"""Compute Mean Squared Error between two tensors using function:

    .. math::
        \operatorname {MSE}\left(Y, \hat{Y}\right) =\frac {1}{n}\sum _{i=1}^{n}\left(y_i-\hat{y_i} \right)^{2}.

    More info: https://en.wikipedia.org/wiki/Mean_squared_error

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return compute_mean_error_metrics(y_pred, y, func=self.sq_func)


class MAEMetric(RegressionMetric):
    r"""Compute Mean Absolute Error between two tensors using function:

    .. math::
        \operatorname {MAE}\left(Y, \hat{Y}\right) =\frac {1}{n}\sum _{i=1}^{n}\left|y_i-\hat{y_i}\right|.

    More info: https://en.wikipedia.org/wiki/Mean_absolute_error

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.abs_func = torch.abs

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return compute_mean_error_metrics(y_pred, y, func=self.abs_func)


class RMSEMetric(RegressionMetric):
    r"""Compute Root Mean Squared Error between two tensors using function:

    .. math::
        \operatorname {RMSE}\left(Y, \hat{Y}\right) ={ \sqrt{ \frac {1}{n}\sum _{i=1}^{n}\left(y_i-\hat{y_i}\right)^2 } } \
        = \sqrt {\operatorname{MSE}\left(Y, \hat{Y}\right)}.

    More info: https://en.wikipedia.org/wiki/Root-mean-square_deviation

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(self, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mse_out = compute_mean_error_metrics(y_pred, y, func=self.sq_func)
        return torch.sqrt(mse_out)


class PSNRMetric(RegressionMetric):
    r"""Compute Peak Signal To Noise Ratio between two tensors using function:

    .. math::
        \operatorname{PSNR}\left(Y, \hat{Y}\right) = 20 \cdot \log_{10} \left({\mathit{MAX}}_Y\right) \
        -10 \cdot \log_{10}\left(\operatorname{MSE\left(Y, \hat{Y}\right)}\right)

    More info: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Help taken from:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py line 4139

    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        max_val: The dynamic range of the images/volumes (i.e., the difference between the
            maximum and the minimum allowed values e.g. 255 for a uint8 image).
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(
        self, max_val: int | float, reduction: MetricReduction | str = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.max_val = max_val
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> Any:
        mse_out = compute_mean_error_metrics(y_pred, y, func=self.sq_func)
        return 20 * math.log10(self.max_val) - 10 * torch.log10(mse_out)


def compute_mean_error_metrics(y_pred: torch.Tensor, y: torch.Tensor, func: Callable) -> torch.Tensor:
    # reducing in only channel + spatial dimensions (not batch)
    # reduction of batch handled inside __call__() using do_metric_reduction() in respective calling class
    flt = partial(torch.flatten, start_dim=1)
    return torch.mean(flt(func(y - y_pred)), dim=-1, keepdim=True)


class KernelType(StrEnum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"


class SSIMMetric(RegressionMetric):
    r"""
    Computes the Structural Similarity Index Measure (SSIM).

    .. math::
        \operatorname {SSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        data_range: value range of input images. (usually 1.0 or 255)
        kernel_type: type of kernel, can be "gaussian" or "uniform".
        win_size: window size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        win_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)

        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(win_size, Sequence):
            win_size = ensure_tuple_rep(win_size, spatial_dims)
        self.kernel_size = win_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
            y: Reference image.
                It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].

        Raises:
            ValueError: when `y_pred` is not a 2D or 3D image.
        """
        dims = y_pred.ndimension()
        if self.spatial_dims == 2 and dims != 4:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width) when using {self.spatial_dims} "
                f"spatial dimensions, got {dims}."
            )

        if self.spatial_dims == 3 and dims != 5:
            raise ValueError(
                f"y_pred should have 4 dimensions (batch, channel, height, width, depth) when using {self.spatial_dims}"
                f" spatial dimensions, got {dims}."
            )

        ssim_value_full_image, _ = compute_ssim_and_cs(
            y_pred=y_pred,
            y=y,
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
        )

        ssim_per_batch: torch.Tensor = ssim_value_full_image.view(ssim_value_full_image.shape[0], -1).mean(
            1, keepdim=True
        )

        return ssim_per_batch


def _gaussian_kernel(
    spatial_dims: int, num_channels: int, kernel_size: Sequence[int], kernel_sigma: Sequence[float]
) -> torch.Tensor:
    """Computes 2D or 3D gaussian kernel.

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        num_channels: number of channels in the image
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
    """

    def gaussian_1d(kernel_size: int, sigma: float) -> torch.Tensor:
        """Computes 1D gaussian kernel.

        Args:
            kernel_size: size of the gaussian kernel
            sigma: Standard deviation of the gaussian kernel
        """
        dist = torch.arange(start=(1 - kernel_size) / 2, end=(1 + kernel_size) / 2, step=1)
        gauss = torch.exp(-torch.pow(dist / sigma, 2) / 2)
        return (gauss / gauss.sum()).unsqueeze(dim=0)

    gaussian_kernel_x = gaussian_1d(kernel_size[0], kernel_sigma[0])
    gaussian_kernel_y = gaussian_1d(kernel_size[1], kernel_sigma[1])
    kernel = torch.matmul(gaussian_kernel_x.t(), gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    kernel_dimensions: tuple[int, ...] = (num_channels, 1, kernel_size[0], kernel_size[1])

    if spatial_dims == 3:
        gaussian_kernel_z = gaussian_1d(kernel_size[2], kernel_sigma[2])[None,]
        kernel = torch.mul(
            kernel.unsqueeze(-1).repeat(1, 1, kernel_size[2]),
            gaussian_kernel_z.expand(kernel_size[0], kernel_size[1], kernel_size[2]),
        )
        kernel_dimensions = (num_channels, 1, kernel_size[0], kernel_size[1], kernel_size[2])

    return kernel.expand(kernel_dimensions)


def compute_ssim_and_cs(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    spatial_dims: int,
    kernel_size: Sequence[int],
    kernel_sigma: Sequence[float],
    data_range: float = 1.0,
    kernel_type: KernelType | str = KernelType.GAUSSIAN,
    k1: float = 0.01,
    k2: float = 0.03,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Function to compute the Structural Similarity Index Measure (SSIM) and Contrast Sensitivity (CS) for a batch
    of images.

    Args:
        y_pred: batch of predicted images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
        y: batch of target images with shape (batch_size, channels, spatial_dim1, spatial_dim2[, spatial_dim3])
        kernel_size: the size of the kernel to use for the SSIM computation.
        kernel_sigma: the standard deviation of the kernel to use for the SSIM computation.
        spatial_dims: number of spatial dimensions of the images (2, 3)
        data_range: the data range of the images.
        kernel_type: the type of kernel to use for the SSIM computation. Can be either "gaussian" or "uniform".
        k1: the first stability constant.
        k2: the second stability constant.

    Returns:
        ssim: the Structural Similarity Index Measure score for the batch of images.
        cs: the Contrast Sensitivity for the batch of images.
    """
    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    y_pred = convert_data_type(y_pred, output_type=torch.Tensor, dtype=torch.float)[0]
    y = convert_data_type(y, output_type=torch.Tensor, dtype=torch.float)[0]

    num_channels = y_pred.size(1)

    if kernel_type == KernelType.GAUSSIAN:
        kernel = _gaussian_kernel(spatial_dims, num_channels, kernel_size, kernel_sigma)
    elif kernel_type == KernelType.UNIFORM:
        kernel = torch.ones((num_channels, 1, *kernel_size)) / torch.prod(torch.tensor(kernel_size))

    kernel = convert_to_dst_type(src=kernel, dst=y_pred)[0]

    c1 = (k1 * data_range) ** 2  # stability constant for luminance
    c2 = (k2 * data_range) ** 2  # stability constant for contrast

    conv_fn = getattr(F, f"conv{spatial_dims}d")
    mu_x = conv_fn(y_pred, kernel, groups=num_channels)
    mu_y = conv_fn(y, kernel, groups=num_channels)
    mu_xx = conv_fn(y_pred * y_pred, kernel, groups=num_channels)
    mu_yy = conv_fn(y * y, kernel, groups=num_channels)
    mu_xy = conv_fn(y_pred * y, kernel, groups=num_channels)

    sigma_x = mu_xx - mu_x * mu_x
    sigma_y = mu_yy - mu_y * mu_y
    sigma_xy = mu_xy - mu_x * mu_y

    contrast_sensitivity = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    ssim_value_full_image = ((2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)) * contrast_sensitivity

    return ssim_value_full_image, contrast_sensitivity


class MultiScaleSSIMMetric(RegressionMetric):
    """
    Computes the Multi-Scale Structural Similarity Index Measure (MS-SSIM).

    MS-SSIM reference paper:
        Wang, Z., Simoncelli, E.P. and Bovik, A.C., 2003, November. "Multiscale structural
        similarity for image quality assessment." In The Thirty-Seventh Asilomar Conference
        on Signals, Systems & Computers, 2003 (Vol. 2, pp. 1398-1402). IEEE

    Args:
        spatial_dims: number of spatial dimensions of the input images.
        data_range: value range of input images. (usually 1.0 or 255)
        kernel_type: type of kernel, can be "gaussian" or "uniform".
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        weights: parameters for image similarity and contrast sensitivity at different resolution scores.
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self,
        spatial_dims: int,
        data_range: float = 1.0,
        kernel_type: KernelType | str = KernelType.GAUSSIAN,
        kernel_size: int | Sequence[int] = 11,
        kernel_sigma: float | Sequence[float] = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)

        self.spatial_dims = spatial_dims
        self.data_range = data_range
        self.kernel_type = kernel_type

        if not isinstance(kernel_size, Sequence):
            kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)
        self.kernel_size = kernel_size

        if not isinstance(kernel_sigma, Sequence):
            kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
        self.kernel_sigma = kernel_sigma

        self.k1 = k1
        self.k2 = k2
        self.weights = weights

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return compute_ms_ssim(
            y_pred=y_pred,
            y=y,
            spatial_dims=self.spatial_dims,
            data_range=self.data_range,
            kernel_type=self.kernel_type,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            k1=self.k1,
            k2=self.k2,
            weights=self.weights,
        )


def compute_ms_ssim(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    spatial_dims: int,
    data_range: float = 1.0,
    kernel_type: KernelType | str = KernelType.GAUSSIAN,
    kernel_size: int | Sequence[int] = 11,
    kernel_sigma: float | Sequence[float] = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    weights: Sequence[float] = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
) -> torch.Tensor:
    """
    Args:
        y_pred: Predicted image.
            It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
        y: Reference image.
            It must be a 2D or 3D batch-first tensor [B,C,H,W] or [B,C,H,W,D].
        spatial_dims: number of spatial dimensions of the input images.
        data_range: value range of input images. (usually 1.0 or 255)
        kernel_type: type of kernel, can be "gaussian" or "uniform".
        kernel_size: size of kernel
        kernel_sigma: standard deviation for Gaussian kernel.
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        weights: parameters for image similarity and contrast sensitivity at different resolution scores.
    Raises:
        ValueError: when `y_pred` is not a 2D or 3D image.
    """
    dims = y_pred.ndimension()
    if spatial_dims == 2 and dims != 4:
        raise ValueError(
            f"y_pred should have 4 dimensions (batch, channel, height, width) when using {spatial_dims} "
            f"spatial dimensions, got {dims}."
        )

    if spatial_dims == 3 and dims != 5:
        raise ValueError(
            f"y_pred should have 4 dimensions (batch, channel, height, width, depth) when using {spatial_dims}"
            f" spatial dimensions, got {dims}."
        )

    if not isinstance(kernel_size, Sequence):
        kernel_size = ensure_tuple_rep(kernel_size, spatial_dims)

    if not isinstance(kernel_sigma, Sequence):
        kernel_sigma = ensure_tuple_rep(kernel_sigma, spatial_dims)
    # check if image have enough size for the number of downsamplings and the size of the kernel
    weights_div = max(1, (len(weights) - 1)) ** 2
    y_pred_spatial_dims = y_pred.shape[2:]
    for i in range(len(y_pred_spatial_dims)):
        if y_pred_spatial_dims[i] // weights_div <= kernel_size[i] - 1:
            raise ValueError(
                f"For a given number of `weights` parameters {len(weights)} and kernel size "
                f"{kernel_size[i]}, the image height must be larger than "
                f"{(kernel_size[i] - 1) * weights_div}."
            )

    weights_tensor = torch.tensor(weights, device=y_pred.device, dtype=torch.float)

    avg_pool = getattr(F, f"avg_pool{spatial_dims}d")

    multiscale_list: list[torch.Tensor] = []
    for _ in range(len(weights_tensor)):
        ssim, cs = compute_ssim_and_cs(
            y_pred=y_pred,
            y=y,
            spatial_dims=spatial_dims,
            data_range=data_range,
            kernel_type=kernel_type,
            kernel_size=kernel_size,
            kernel_sigma=kernel_sigma,
            k1=k1,
            k2=k2,
        )

        cs_per_batch = cs.view(cs.shape[0], -1).mean(1)

        multiscale_list.append(torch.relu(cs_per_batch))
        y_pred = avg_pool(y_pred, kernel_size=2)
        y = avg_pool(y, kernel_size=2)

    ssim = ssim.view(ssim.shape[0], -1).mean(1)
    multiscale_list[-1] = torch.relu(ssim)
    multiscale_list_tensor = torch.stack(multiscale_list)

    ms_ssim_value_full_image = torch.prod(multiscale_list_tensor ** weights_tensor.view(-1, 1), dim=0)

    ms_ssim_per_batch: torch.Tensor = ms_ssim_value_full_image.view(ms_ssim_value_full_image.shape[0], -1).mean(
        1, keepdim=True
    )

    return ms_ssim_per_batch
