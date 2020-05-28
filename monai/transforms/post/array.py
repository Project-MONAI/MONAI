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
        to_onehot (bool): whether to convert the data to One-Hot format first, default is False.
        num_classes (int): the class number used to convert to One-Hot format if `to_onehot` is True.

    """

    def __init__(self, to_onehot=False, num_classes=None):
        self.to_onehot = to_onehot
        self.num_classes = num_classes

    def __call__(self, img, to_onehot=None, num_classes=None):
        if to_onehot or self.to_onehot:
            if num_classes is None:
                num_classes = self.num_classes
            assert isinstance(num_classes, int), "must specify class number for One-Hot."
            img = one_hot(img, num_classes)
        n_classes = img.shape[1]
        outputs = list()
        for i in range(n_classes):
            outputs.append(img[:, i : i + 1])

        return outputs


class Activations(Transform):
    """
    Add activation operations to the model output, typically `Sigmoid` or `Softmax`.

    Args:
        sigmoid (bool): whether to execute sigmoid function on model output before transform.
        softmax (bool): whether to execute softmax function on model output before transform.
        other (Callable): callable function to execute other activation layers, for example:
            `other = lambda x: torch.tanh(x)`

    """

    def __init__(self, sigmoid=False, softmax=False, other=None):
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other = other

    def __call__(self, img, sigmoid=None, softmax=None, other=None):
        if sigmoid is True and softmax is True:
            raise ValueError("sigmoid=True and softmax=True are not compatible.")
        if sigmoid or self.sigmoid:
            img = torch.sigmoid(img)
        if softmax or self.softmax:
            img = torch.softmax(img, dim=1)
        act_func = self.other if other is None else other
        if act_func is not None:
            if not callable(act_func):
                raise ValueError("act_func must be a Callable function.")
            img = act_func(img)

        return img


class AsDiscrete(Transform):
    """Execute after model forward to transform model output to discrete values.
    It can complete below operations:
        #. execute `argmax` for input logits values.
        #. threshold input value to 0.0 or 1.0.
        #. convert input value to One-Hot format

    Args:
        argmax (bool): whether to execute argmax function on input data before transform.
        to_onehot (bool): whether to convert input data into the one-hot format. Defaults to False.
        n_classes (bool): the number of classes to convert to One-Hot format.
        threshold_values (bool): whether threshold the float value to int number 0 or 1, default is False.
        logit_thresh (float): the threshold value for thresholding operation, default is 0.5.

    """

    def __init__(self, argmax=False, to_onehot=False, n_classes=None, threshold_values=False, logit_thresh=0.5):
        self.argmax = argmax
        self.to_onehot = to_onehot
        self.n_classes = n_classes
        self.threshold_values = threshold_values
        self.logit_thresh = logit_thresh

    def __call__(self, img, argmax=None, to_onehot=None, n_classes=None, threshold_values=None, logit_thresh=None):
        if argmax or self.argmax:
            img = torch.argmax(img, dim=1, keepdim=True)

        if to_onehot or self.to_onehot:
            img = one_hot(img, self.n_classes if n_classes is None else n_classes)

        if threshold_values or self.threshold_values:
            img = img >= (self.logit_thresh if logit_thresh is None else logit_thresh)

        return img.float()
