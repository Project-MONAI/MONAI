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

from typing import Callable, Union

from monai.handlers.ignite_metric import IgniteMetric
from monai.metrics import MAEMetric, MSEMetric, PSNRMetric, RMSEMetric
from monai.utils import MetricReduction


class MeanSquaredError(IgniteMetric):
    """
    Computes Mean Squared Error from full size Tensor and collects average over batch, iterations.
    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

        Args:
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: mean squared error of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        See also:
            :py:class:`monai.metrics.MSEMetric`
        """
        metric_fn = MSEMetric(reduction=reduction)
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)


class MeanAbsoluteError(IgniteMetric):
    """
    Computes Mean Absolute Error from full size Tensor and collects average over batch, iterations.
    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

        Args:
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: mean squared error of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        See also:
            :py:class:`monai.metrics.MAEMetric`
        """
        metric_fn = MAEMetric(reduction=reduction)
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)


class RootMeanSquaredError(IgniteMetric):
    """
    Computes Root Mean Squared Error from full size Tensor and collects average over batch, iterations.
    """

    def __init__(
        self,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

        Args:
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: mean squared error of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.

        See also:
            :py:class:`monai.metrics.RMSEMetric`
        """
        metric_fn = RMSEMetric(reduction=reduction)
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)


class PeakSignalToNoiseRatio(IgniteMetric):
    """
    Computes Peak Signal to Noise Ratio from full size Tensor and collects average over batch, iterations.
    """

    def __init__(
        self,
        max_val: Union[int, float],
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
        output_transform: Callable = lambda x: x,
        save_details: bool = True,
    ) -> None:
        """

        Args:
            max_val: The dynamic range of the images/volumes (i.e., the difference between the
                maximum and the minimum allowed values e.g. 255 for a uint8 image).
            reduction: define the mode to reduce metrics, will only execute reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
            output_transform: callable to extract `y_pred` and `y` from `ignite.engine.state.output` then
                construct `(y_pred, y)` pair, where `y_pred` and `y` can be `batch-first` Tensors or
                lists of `channel-first` Tensors. the form of `(y_pred, y)` is required by the `update()`.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            save_details: whether to save metric computation details per image, for example: mean squared error of every image.
                default to True, will save to `engine.state.metric_details` dict with the metric name as key.
            reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,

        See also:
            :py:class:`monai.metrics.PSNRMetric`
        """
        metric_fn = PSNRMetric(max_val=max_val, reduction=reduction)
        super().__init__(metric_fn=metric_fn, output_transform=output_transform, save_details=save_details)
