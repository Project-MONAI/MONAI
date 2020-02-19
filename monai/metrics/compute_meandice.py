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

import torch

from monai.networks.utils import one_hot


def compute_meandice(y_pred,
                     y,
                     include_background=False,
                     to_onehot_y=True,
                     mutually_exclusive=True,
                     add_sigmoid=False,
                     logit_thresh=None):
    """Computes dice score metric from full size Tensor and collects average.

    Args:
        y_pred (torch.Tensor): input data to compute, typical segmentation model output.
                               it must be One-Hot format and first dim is batch, example shape: [16, 3, 32, 32].
        y (torch.Tensor): ground truth to compute mean dice metric, the first dim is batch.
                          example shape: [16, 3, 32, 32] for 3-class one-hot labels.
                          alternative shape: [16, 1, 32, 32] and set `to_onehot_y=True` to convert it into [16, 3, 32, 32].
        include_background (Bool): whether to skip dice computation on the first channel of the predicted output.
        to_onehot_y (Bool): whether to convert `y` into the one-hot format.
        mutually_exclusive (Bool): if True, `y_pred` will be converted into a binary matrix using
            a combination of argmax and to_onehot.
        add_sigmoid (Bool): whether to add sigmoid function to y_pred before computation.
        logit_thresh (Float): the threshold value used to convert `y_pred` into a binary matrix.

    Note:
        This method provide two options to convert `y_pred` into a binary matrix:
            (1) when `mutually_exclusive` is True, it uses a combination of argmax and to_onehot
            (2) when `mutually_exclusive` is False, it uses a threshold `logit_thresh`
                (optionally with a sigmoid function before thresholding).

    """
    n_channels_y_pred = y_pred.shape[1]

    if mutually_exclusive:
        if logit_thresh is not None or add_sigmoid:
            raise ValueError('`logit_thresh` and `add_sigmoid` are incompatible when mutually_exclusive is True.')
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        y_pred = one_hot(y_pred, n_channels_y_pred)
    else:  # channel-wise thresholding
        if add_sigmoid:
            y_pred = torch.sigmoid(y_pred)
        if logit_thresh is not None:
            y_pred = (y_pred >= logit_thresh).float()

    if to_onehot_y:
        y = one_hot(y, n_channels_y_pred)

    if not include_background:
        y = y[:, 1:] if y.shape[1] > 1 else y
        y_pred = y_pred[:, 1:] if y_pred.shape[1] > 1 else y_pred

    assert y.shape == y_pred.shape, ("Ground truth one-hot has differing shape (%r) from source (%r)" %
                                     (y.shape, y_pred.shape))

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis = list(range(2, y_pred.dim()))
    intersection = torch.sum(y * y_pred, reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, reduce_axis)
    denominator = y_o + y_pred_o

    f = (2.0 * intersection) / denominator
    # final reduce_mean across batches and channels
    return torch.mean(f)
