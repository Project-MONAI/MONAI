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
"""
A collection of "vanilla" transforms for the model output tensors
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import torch
from monai.transforms.compose import Transform
from monai.networks.utils import one_hot


class SplitChannel(Transform):
    """
    Split PyTorch Tensor data according to the channel dim, if only 1 channel, convert to One-Hot
    format first based on the class number. Users can use this transform to compute metrics on every
    single class to get more details of validation/evaluation. Expected input shape:
    (batch_size, num_channels, spatial_dim_1[, spatial_dim_2, ...])

    Args:
        to_onehot (bool): whether to convert the data to One-Hot format, default is False.
        num_classes (int): the class number used to convert to One-Hot format if `to_onehot` is True.

    """

    def __init__(self, to_onehot=False, num_classes=None):
        self.to_onehot = to_onehot
        self.num_classes = num_classes

    def __call__(self, img, to_onehot=None, num_classes=None):
        if self.to_onehot if to_onehot is None else to_onehot:
            if num_classes is None:
                num_classes = self.num_classes
            assert isinstance(num_classes, int), "must specify class number for One-Hot."
            img = one_hot(img, num_classes)
        n_classes = img.shape[1]
        outputs = list()
        for i in range(n_classes):
            outputs.append(img[:, i : i + 1])

        return outputs


class ConvertForMetrics(Transform):
    """Execute after model forward to transform model output and labels for Ignite metrics.
    It can complete below operations:
        #. add `sigmoid` or `softmax` to y_pred
        #. do `argmax` for y_pred
        #. round y_pred value to 0.0 or 1.0
        #. convert y_pred or y to One-Hot format

    Args:
        add_sigmoid (bool): whether to add sigmoid function to y_pred before transform.
        add_softmax (bool): whether to add softmax function to y_pred before transform.
        add_argmax (bool): whether to add argmax function to y_pred before transform.
        to_onehot_y_pred (bool): whether to convert `y_pred` into the one-hot format. Defaults to False.
        to_onehot_y (bool): whether to convert `y` into the one-hot format. Defaults to False.
        n_classes (bool): the number of classes to convert to One-Hot format, if None, use `y_pred.shape[1]`
        round_values (bool): whether round the value to 0 and 1, default is False.
        logit_thresh (float): the threshold value to round value to 0.0 and 1.0, default is 0.5.

    """

    def __init__(
        self,
        add_sigmoid=False,
        add_softmax=False,
        add_argmax=False,
        to_onehot_y_pred=False,
        to_onehot_y=False,
        n_classes=None,
        round_values=False,
        logit_thresh=0.5,
    ):
        self.add_sigmoid = add_sigmoid
        self.add_softmax = add_softmax
        self.add_argmax = add_argmax
        self.to_onehot_y_pred = to_onehot_y_pred
        self.to_onehot_y = to_onehot_y
        self.n_classes = n_classes
        self.round_values = round_values
        self.logit_thresh = logit_thresh

    def __call__(self, y_pred, y=None):
        """
        Args:
            y_pred (Tensor): model output data, expected shape: [B, C, spatial_dims(0 - N)].
            y (Tensor): label data, can be None if do inference.

        """
        n_classes = y_pred.shape[1] if self.n_classes is None else self.n_classes
        if self.add_sigmoid is True:
            y_pred = torch.sigmoid(y_pred)
        if self.add_softmax is True:
            y_pred = torch.softmax(y_pred, dim=1)
        if self.add_argmax is True:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)

        if self.to_onehot_y_pred:
            y_pred = one_hot(y_pred, n_classes)
        if self.to_onehot_y and y is not None:
            y = one_hot(y, n_classes)
        if self.round_values:
            y_pred = y_pred >= self.logit_thresh

        return y_pred.float(), y.float() if y is not None else None
