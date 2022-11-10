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

import math
from abc import abstractmethod
from functools import partial
from typing import Any, Union

import torch

from monai.losses.ssim_loss import SSIMLoss
from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction

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

    def __init__(
        self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):
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

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
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

    def __init__(
        self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

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

    def __init__(
        self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.abs_func = torch.abs

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

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

    def __init__(
        self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN, get_not_nans: bool = False
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_pred = y_pred.float()
        y = y.float()

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
        self,
        max_val: Union[int, float],
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.max_val = max_val
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y: torch.Tensor) -> Any:
        y_pred = y_pred.float()
        y = y.float()

        mse_out = compute_mean_error_metrics(y_pred, y, func=self.sq_func)
        return 20 * math.log10(self.max_val) - 10 * torch.log10(mse_out)


def compute_mean_error_metrics(y_pred: torch.Tensor, y: torch.Tensor, func) -> torch.Tensor:
    # reducing in only channel + spatial dimensions (not batch)
    # reduction of batch handled inside __call__() using do_metric_reduction() in respective calling class
    flt = partial(torch.flatten, start_dim=1)
    return torch.mean(flt(func(y - y_pred)), dim=-1, keepdim=True)


class SSIMMetric(RegressionMetric):
    r"""
    Build a Pytorch version of the SSIM metric based on the original formula of SSIM

    .. math::
        \operatorname {SSIM}(x,y) =\frac {(2 \mu_x \mu_y + c_1)(2 \sigma_{xy} + c_2)}{((\mu_x^2 + \
                \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}

    For more info, visit
        https://vicuesoft.com/glossary/term/ssim-ms-ssim/

    Modified and adopted from:
        https://github.com/facebookresearch/fastMRI/blob/main/banding_removal/fastmri/ssim_loss_mixin.py

    SSIM reference paper:
        Wang, Zhou, et al. "Image quality assessment: from error visibility to structural
        similarity." IEEE transactions on image processing 13.4 (2004): 600-612.

    Args:
        data_range: dynamic range of the data
        win_size: gaussian weighting window size
        k1: stability constant used in the luminance denominator
        k2: stability constant used in the contrast denominator
        spatial_dims: if 2, input shape is expected to be (B,C,W,H). if 3, it is expected to be (B,C,W,H,D)
        reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans)
    """

    def __init__(
        self, data_range: torch.Tensor, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, spatial_dims: int = 2,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN, get_not_nans: bool = False
    ):
        super().__init__(reduction=reduction, get_not_nans=get_not_nans)
        self.data_range = data_range
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.spatial_dims = spatial_dims

    def _compute_metric(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: first sample (e.g., the reference image). Its shape is (B,C,W,H) for 2D data and (B,C,W,H,D) for 3D.
                A fastMRI sample should use the 2D format with C being the number of slices.
            y: second sample (e.g., the reconstructed image). It has similar shape as x

        Returns:
            ssim_value

        Example:
            .. code-block:: python

                import torch
                x = torch.ones([1,1,10,10])/2 # ground truth
                y = torch.ones([1,1,10,10])/2 # prediction
                data_range = x.max().unsqueeze(0)
                # the following line should print 1.0 (or 0.9999)
                print(SSIMMetric(data_range=data_range,spatial_dims=2)._compute_metric(x,y))
        """
        ssim_value = torch.empty((1), dtype=torch.float)
        if x.shape[0]==1:
            ssim_value: torch.Tensor = 1 - SSIMLoss(self.win_size, self.k1, self.k2, self.spatial_dims)(
                x, y, self.data_range
            )
        elif x.shape[0]>1:

            for i in range(x.shape[0]):
                ssim_val: torch.Tensor = 1 - SSIMLoss(self.win_size, self.k1, self.k2, self.spatial_dims)(
                    x[i:i+1], y[i:i+1], self.data_range
                )
                if i == 0:
                    ssim_value = ssim_val
                else:
                    ssim_value = torch.cat((ssim_value.view(1), ssim_val.view(1)), dim=0)

            ssim_value = ssim_value.view(-1,1)
        else:
            raise ValueError("Batch size is not nonnegative integer value")
