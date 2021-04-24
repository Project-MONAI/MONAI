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
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Union

import torch

from monai.metrics.utils import do_metric_reduction
from monai.utils import MetricReduction


class RegressionMetric(ABC):
    def __init__(self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN) -> None:
        super().__init__()
        self.reduction = reduction

    def _reduce(self, f: torch.Tensor):
        return do_metric_reduction(f, self.reduction)

    def _check_shape(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> None:
        if y_pred.shape != y_target.shape:
            raise ValueError(
                "y_pred and y_target shapes dont match, received y_pred: [{}] and y_target: [{}]".format(
                    y_pred.shape, y_target.shape
                )
            )

        # also check if there is atleast one non-batch dimension i.e. num_dims >= 2
        if len(y_pred.shape) < 2:
            raise ValueError("either channel or spatial dimensions required, found only batch dimension")

    @abstractmethod
    def _compute_metric(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def __call__(self, y_pred: torch.Tensor, y_target: torch.Tensor):
        self._check_shape(y_pred, y_target)
        out = self._compute_metric(y_pred, y_target)
        y, not_nans = self._reduce(out)
        return y


class MSEMetric(RegressionMetric):
    def __init__(self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN) -> None:
        super().__init__(reduction=reduction)
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # https://en.wikipedia.org/wiki/Mean_squared_error
        # Implments equation: {\displaystyle \operatorname {MSE} ={\frac {1}{n}}\sum _{i=1}^{n}(Y_{i}-{\hat {Y_{i}}})^{2}.}

        y_pred = y_pred.float()
        y_target = y_target.float()

        mse_out = compute_mean_error_metrics(y_pred, y_target, func=self.sq_func)

        return mse_out


class MAEMetric(RegressionMetric):
    def __init__(self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN) -> None:
        super().__init__(reduction=reduction)
        self.abs_func = torch.abs

    def _compute_metric(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # https://en.wikipedia.org/wiki/Mean_absolute_error
        # Implments equation: {\displaystyle \mathrm {MAE} ={\frac {\sum _{i=1}^{n}\left|y_{i}-x_{i}\right|}{n}}.}

        y_pred = y_pred.float()
        y_target = y_target.float()

        mae_out = compute_mean_error_metrics(y_pred, y_target, func=self.abs_func)

        return mae_out


class RMSEMetric(RegressionMetric):
    def __init__(self, reduction: Union[MetricReduction, str] = MetricReduction.MEAN) -> None:
        super().__init__(reduction=reduction)
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # https://en.wikipedia.org/wiki/Root-mean-square_deviation
        # Implments equation: {\displaystyle \operatorname {RMSD} ={\sqrt {\frac {\sum _{t=1}^{T}({\hat {y}}_{t}-y_{t})^{2}}{T}}}.}

        y_pred = y_pred.float()
        y_target = y_target.float()

        mse_out = compute_mean_error_metrics(y_pred, y_target, func=self.sq_func)

        # https://en.wikipedia.org/wiki/Root-mean-square_deviation#Formula
        rmse_out = torch.sqrt(mse_out)

        return rmse_out


class PSNRMetric(RegressionMetric):
    def __init__(
        self, max_val: Union[int, float], reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ) -> None:
        super().__init__(reduction=reduction)
        self.max_val = max_val
        self.sq_func = partial(torch.pow, exponent=2.0)

    def _compute_metric(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> Any:
        # https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        # Implments equation: \mathit{PSNR} = 20 \cdot \log_{10} \left({\mathit{MAX}}_{I}\right) -10 \cdot \log_{10}\left(\mathit{MSE}\right)
        # Help from: https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/ops/image_ops_impl.py#L3401

        y_pred = y_pred.float()
        y_target = y_target.float()

        mse_out = compute_mean_error_metrics(y_pred, y_target, func=self.sq_func)

        # \mathit{PSNR} = 20 \cdot \log_{10} \left({\mathit{MAX}}_{I}\right) -10 \cdot \log_{10}\left(\mathit{MSE}\right)
        psnr_val = 20 * math.log10(self.max_val) - 10 * torch.log10(mse_out)

        return psnr_val


def compute_mean_error_metrics(y_pred: torch.Tensor, y_target: torch.Tensor, func) -> torch.Tensor:
    # reducing only channel + spatial dimensions (not batch)
    # reducion batch handled inside __call__() using do_metric_reduction() in respective calling class
    flt = partial(torch.flatten, start_dim=1)
    error_metric = torch.mean(flt(func(y_target - y_pred)), dim=-1, keepdim=True)

    return error_metric
