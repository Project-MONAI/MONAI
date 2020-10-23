# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Callable, Optional, Sequence, Union

import torch

from monai.networks import one_hot
from monai.utils import MetricReduction


def do_onehot(input_data: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    This function is used to convert the input data into one-hot fotmat.
    Args:
        input_data: the input that to be converted, the first dim is batch.
        num_classes: the number of classes.

    """
    if num_classes == 1:
        warnings.warn("y_pred has only one channel, to_onehot_y=True ignored.")
    else:
        input_data = one_hot(input_data, num_classes=num_classes)
    return input_data


def do_activation(input_data: torch.Tensor, activation: Union[str, Callable] = "softmax") -> torch.Tensor:
    """
    This function is used to do activation for the input data.

    Args:
        input_data: the input that to be activated, the first dim is batch.
        activation: can be ``"sigmoid"`` or ``"softmax"``, or a callable function. Defaults to ``"softmax"``.
            An example for callable function: ``activation = lambda x: torch.log_softmax(x)``.

    Raises:
        NotImplementedError: When input an activation name that is not implemented.
    """
    input_ndim = input_data.ndimension()
    if activation == "softmax":
        if input_ndim == 1:
            warnings.warn("input_data has only one channel, softmax ignored.")
        else:
            input_data = input_data.float().softmax(dim=1)
    elif activation == "sigmoid":
        input_data = input_data.float().sigmoid()
    elif callable(activation):
        input_data = activation(input_data)
    else:
        raise NotImplementedError("activation can only be sigmoid, softmax or a callable function.")
    return input_data


def do_binarization(
    input_data: torch.Tensor,
    bin_mode: str = "threshold",
    bin_threshold: Union[float, Sequence[float]] = 0.5,
) -> torch.Tensor:
    """
    Args:
        input_data: the input that to be binarized, in the shape [B] or [BN] or [BNHW] or [BNHWD].
        bin_mode: can be ``"threshold"`` or ``"mutually_exclusive"``, or a callable function.
            - ``"threshold"``, a single threshold or a sequence of thresholds should be set.
            - ``"mutually_exclusive"``, `input_data` will be converted by a combination of
            argmax and to_onehot.
        bin_threshold: the threshold to binarize the input data, can be a single value or a sequence of
            values that each one of the value represents a threshold for a class.

    Raises:
        AssertionError: when `bin_threshold` is a sequence and the input has the shape [B].
        AssertionError: when `bin_threshold` is a sequence but the length != the number of classes.
        AssertionError: when `bin_mode` is ``"mutually_exclusive"`` the input has the shape [B].
        AssertionError: when `bin_mode` is ``"mutually_exclusive"`` the input has the shape [B, 1].
    """
    input_ndim = input_data.ndimension()
    if bin_mode == "threshold":
        if isinstance(bin_threshold, Sequence):
            assert input_ndim > 1, "a sequence of thresholds are used for multi-class tasks."
            error_hint = "the length of the sequence should be the same as the number of classes."
            n_classes = input_data.shape[1]
            assert n_classes == len(bin_threshold), "{}".format(error_hint)
            for cls_num in range(n_classes):
                input_data[:, cls_num] = (input_data[:, cls_num] >= bin_threshold[cls_num]).float()
        else:
            input_data = (input_data >= bin_threshold).float()
    elif bin_mode == "mutually_exclusive":
        assert input_ndim > 1, "mutually_exclusive is used for multi-class tasks."
        n_classes = input_data.shape[1]
        assert n_classes > 1, "mutually_exclusive is used for multi-class tasks."
        input_data = torch.argmax(input_data, dim=1, keepdim=True)
        input_data = one_hot(input_data, num_classes=n_classes)
    return input_data


def preprocess_input(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    to_onehot_y: bool = False,
    activation: Optional[Union[str, Callable]] = None,
    bin_mode: Optional[str] = None,
    bin_threshold: Union[float, Sequence[float]] = 0.5,
    include_background: bool = True,
):
    """
    Args:
        y_pred: predictions. As for classification tasks,
            `y_pred` should has the shape [BN] where N is larger than 1. As for segmentation tasks,
            the shape should be [BNHW] or [BNHWD].
        y: ground truth, the first dim is batch.
        to_onehot_y: whether to convert `y` into the one-hot format. If not to convert the format, ``y``
        should has the same shape as ``y_pred``. Defaults to False.
        activation: [``None``, ``"sigmoid"``, ``"softmax"``]
            Activation method, if specified, an activation function will be employed for `y_pred` before
            computation. Defaults to None. The parameter can also be a callable function, for example:
            ``activation = lambda x: torch.log_softmax(x)``.
        bin_mode: [``None``, ``"threshold"``, ``"mutually_exclusive"``]
            Binarization method, if specified, a binarization manipulation will be employed for `y_pred` before
            computation. Defaults to None.

            - ``"threshold"``, a single threshold or a sequence of thresholds should be set.
            - ``"mutually_exclusive"``, `y_pred` will be converted by a combination of `argmax` and `to_onehot`.
        bin_threshold: the threshold for binarization, can be a single value or a sequence of
            values that each one of the value represents a threshold for a class.
        include_background: whether to include computation on the first channel of the predicted output.
            Defaults to True.

    Raises:
        AssertionError: When ``y_pred`` only has one dimebsion.
        AssertionError: When ``y`` and ``y_pred`` have different shapes before output.

    """

    # convert y into one-hot version if needed
    assert y_pred.ndimension() > 1, "y_pred should at least has two dimensions batch and n_classes."
    num_classes = y_pred.shape[1]
    if to_onehot_y:
        y = do_onehot(y, num_classes)
    # do activation for y_pred if needed
    if activation is not None:
        y_pred = do_activation(y_pred, activation=activation)
    # binarization for y_pred if needed
    if bin_mode is not None:
        y_pred = do_binarization(y_pred, bin_mode=bin_mode, bin_threshold=bin_threshold)
    # remove background if needed
    if not include_background:
        y = y[:, 1:] if y.shape[1] > 1 else y
        y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred
    assert y.shape == y_pred.shape, "data shapes of y_pred and y do not match."
    y = y.float()
    y_pred = y_pred.float()
    return y_pred, y


def do_metric_reduction(
    f: torch.Tensor,
    reduction: Union[MetricReduction, str] = MetricReduction.MEAN,
):
    """
    This function is to do the metric reduction for calculated metrics of each example's each class.
    Args:
        f: a tensor that contains the calculated metric scores per batch and
            per class. The first two dims should be batch and class.
        reduction: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
        ``"mean_channel"``, ``"sum_channel"``}
        Define the mode to reduce computation result of 1 batch data. Defaults to ``"mean"``.

    Raises:
        ValueError: When ``reduction`` is not one of
            ["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].
    """

    nans = torch.isnan(f)
    not_nans = (~nans).float()
    f[nans] = 0

    t_zero = torch.zeros(1, device=f.device, dtype=torch.float)
    reduction = MetricReduction(reduction)

    if reduction == MetricReduction.MEAN:
        # 2 steps, first, mean by channel (accounting for nans), then by batch
        not_nans = not_nans.sum(dim=1)
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average

        not_nans = (not_nans > 0).float().sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average

    elif reduction == MetricReduction.SUM:
        not_nans = not_nans.sum(dim=0)
        f = torch.sum(f, dim=[0, 1])  # sum over the batch and channel dims
    elif reduction == MetricReduction.MEAN_BATCH:
        not_nans = not_nans.sum(dim=0)
        f = torch.where(not_nans > 0, f.sum(dim=0) / not_nans, t_zero)  # batch average
    elif reduction == MetricReduction.SUM_BATCH:
        not_nans = not_nans.sum(dim=0)
        f = f.sum(dim=0)  # the batch sum
    elif reduction == MetricReduction.MEAN_CHANNEL:
        not_nans = not_nans.sum(dim=1)
        f = torch.where(not_nans > 0, f.sum(dim=1) / not_nans, t_zero)  # channel average
    elif reduction == MetricReduction.SUM_CHANNEL:
        not_nans = not_nans.sum(dim=1)
        f = f.sum(dim=1)  # the channel sum
    elif reduction == MetricReduction.NONE:
        pass
    else:
        raise ValueError(
            f"Unsupported reduction: {reduction}, available options are "
            '["mean", "sum", "mean_batch", "sum_batch", "mean_channel", "sum_channel" "none"].'
        )
    return f
