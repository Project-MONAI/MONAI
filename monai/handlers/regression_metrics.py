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

from typing import Callable, Union

import torch

from monai.handlers.iteration_metric import IterationMetric
from monai.metrics import MAEMetric, MSEMetric, PSNRMetric, RMSEMetric
from monai.utils import MetricReduction


class MeanSquaredError(IterationMetric):
    """
    Computes Mean Squared Error from full size Tensor and collects average over batch, iterations.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = "cpu",
        save_details: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        """

        Args:
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.
            save_details: whether to save metric computation details per image, for example: mean squared error of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.
            reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}
                Define the mode to reduce computation result. Defaults to ``"mean"``.

        See also:
            :py:class:`monai.metrics.MSEMetric`
        """
        metric_fn = MSEMetric(reduction=reduction)
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            device=device,
            save_details=save_details,
        )


class MeanAbsoluteError(IterationMetric):
    """
    Computes Mean Absolute Error from full size Tensor and collects average over batch, iterations.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = "cpu",
        save_details: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        """

        Args:
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.
            save_details: whether to save metric computation details per image, for example: mean absolute error of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.
            reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}
                Define the mode to reduce computation result. Defaults to ``"mean"``.

        See also:
            :py:class:`monai.metrics.MAEMetric`
        """
        metric_fn = MAEMetric(reduction=reduction)
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            device=device,
            save_details=save_details,
        )


class RootMeanSquaredError(IterationMetric):
    """
    Computes Root Mean Squared Error from full size Tensor and collects average over batch, iterations.
    """

    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = "cpu",
        save_details: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        """

        Args:
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.
            save_details: whether to save metric computation details per image, for example: root mean squared error of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.
            reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}
                Define the mode to reduce computation result. Defaults to ``"mean"``.

        See also:
            :py:class:`monai.metrics.RMSEMetric`
        """
        metric_fn = RMSEMetric(reduction=reduction)
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            device=device,
            save_details=save_details,
        )


class PeakSignalToNoiseRatio(IterationMetric):
    """
    Computes Peak Signal to Noise Ratio from full size Tensor and collects average over batch, iterations.
    """

    def __init__(
        self,
        max_val: Union[int, float],
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = "cpu",
        save_details: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
    ) -> None:
        """

        Args:
            max_val: The dynamic range of the images/volumes (i.e., the difference between the
                maximum and the minimum allowed values e.g. 255 for a uint8 image).
            output_transform: transform the ignite.engine.state.output into [y_pred, y] pair.
            device: device specification in case of distributed computation usage.
            save_details: whether to save metric computation details per image, for example: PSNR of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.
            reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}
                Define the mode to reduce computation result. Defaults to ``"mean"``.

        See also:
            :py:class:`monai.metrics.PSNRMetric`
        """
        metric_fn = PSNRMetric(max_val=max_val, reduction=reduction)
        super().__init__(
            metric_fn=metric_fn,
            output_transform=output_transform,
            device=device,
            save_details=save_details,
        )
