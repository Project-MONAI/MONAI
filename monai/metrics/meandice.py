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

import torch

from monai.networks.utils import one_hot


def compute_meandice(
    y_pred: torch.Tensor,
    y: torch.Tensor,
    include_background: bool = True,
    to_onehot_y: bool = False,
    mutually_exclusive: bool = False,
    sigmoid: bool = False,
    logit_thresh: float = 0.5,
):
    """Computes Dice score metric from full size Tensor and collects average.

    Args:
        y_pred (torch.Tensor): input data to compute, typical segmentation model output.
            it must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32].
        y (torch.Tensor): ground truth to compute mean dice metric, the first dim is batch.
            example shape: [16, 1, 32, 32] will be converted into [16, 3, 32, 32].
            alternative shape: [16, 3, 32, 32] and set `to_onehot_y=False` to use 3-class labels directly.
        include_background: whether to skip Dice computation on the first channel of
            the predicted output. Defaults to True.
        to_onehot_y: whether to convert `y` into the one-hot format. Defaults to False.
        mutually_exclusive: if True, `y_pred` will be converted into a binary matrix using
            a combination of argmax and to_onehot.  Defaults to False.
        sigmoid: whether to add sigmoid function to y_pred before computation. Defaults to False.
        logit_thresh (Float): the threshold value used to convert (after sigmoid if `sigmoid=True`)
            `y_pred` into a binary matrix. Defaults to 0.5.

    Returns:
        Dice scores per batch and per class, (shape [batch_size, n_classes]).

    Note:
        This method provides two options to convert `y_pred` into a binary matrix
            (1) when `mutually_exclusive` is True, it uses a combination of ``argmax`` and ``to_onehot``,
            (2) when `mutually_exclusive` is False, it uses a threshold ``logit_thresh``
                (optionally with a ``sigmoid`` function before thresholding).

    """
    n_classes = y_pred.shape[1]
    n_len = len(y_pred.shape)

    if sigmoid:
        y_pred = y_pred.float().sigmoid()

    if n_classes == 1:
        if mutually_exclusive:
            warnings.warn("y_pred has only one class, mutually_exclusive=True ignored.")
        if to_onehot_y:
            warnings.warn("y_pred has only one channel, to_onehot_y=True ignored.")
        if not include_background:
            warnings.warn("y_pred has only one channel, include_background=False ignored.")
        # make both y and y_pred binary
        y_pred = (y_pred >= logit_thresh).float()
        y = (y > 0).float()
    else:  # multi-channel y_pred
        # make both y and y_pred binary
        if mutually_exclusive:
            if sigmoid:
                raise ValueError("sigmoid=True is incompatible with mutually_exclusive=True.")
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            y_pred = one_hot(y_pred, n_classes)
        else:
            y_pred = (y_pred >= logit_thresh).float()
        if to_onehot_y:
            y = one_hot(y, n_classes)

    if not include_background:
        y = y[:, 1:] if y.shape[1] > 1 else y
        y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred

    assert y.shape == y_pred.shape, "Ground truth one-hot has differing shape (%r) from source (%r)" % (
        y.shape,
        y_pred.shape,
    )

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, reduce_axis)
    denominator = y_o + y_pred_o

    f = torch.where(y_o > 0, (2.0 * intersection) / denominator, torch.tensor(float("nan")).to(y_o.float()))
    return f  # returns array of Dice shape: [Batch, n_classes]
