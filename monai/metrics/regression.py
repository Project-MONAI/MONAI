# Copyright 2020 - 2021 MONAI Consortium
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

from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric


class RegressionMetric(CumulativeIterationMetric):
    """
    Base class for regression metrics.
    Input `y_pred` is compared with ground truth `y`.
    Both `y_pred` and `y` are expected to be real-valued, where `y_pred` is output from a regression model.
    `y_preds` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result. Defaults to ``"mean"``.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.

    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def aggregate(self):  # type: ignore
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        f, not_nans = do_metric_reduction(data, self.reduction)
        return (f, not_nans) if self.get_not_nans else f

    def _check_shape(self, y_pred: torch.Tensor, y: torch.Tensor) -> None:
        if y_pred.shape != y.shape:
            raise ValueError(
                "y_pred and y shapes dont match, received y_pred: [{}] and y: [{}]".format(y_pred.shape, y.shape)
            )

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

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
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

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
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

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).

    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        get_not_nans: bool = False,
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

    Args:
        max_val: The dynamic range of the images/volumes (i.e., the difference between the
            maximum and the minimum allowed values e.g. 255 for a uint8 image).
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}
            Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.
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
