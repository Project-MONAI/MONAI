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

from typing import Optional, Callable

import torch
import torch.nn.functional as F

from monai.transforms.compose import Transform
from monai.networks import one_hot
from monai.transforms.utils import get_largest_connected_component_mask
from monai.utils import ensure_tuple


class SplitChannel(Transform):
    """
    Split PyTorch Tensor data according to the channel dim, if only 1 channel, convert to One-Hot
    format first based on the class number. Users can use this transform to compute metrics on every
    single class to get more details of validation/evaluation. Expected input shape:
    ``(batch_size, num_channels, [spatial_dim_1, spatial_dim_2, ...])``

    Args:
        to_onehot: whether to convert the data to One-Hot format first.
            Defaults to ``False``.
        num_classes: the class number used to convert to One-Hot format if `to_onehot` is True.
            Defaults to ``None``.
    """

    def __init__(self, to_onehot: bool = False, num_classes: Optional[int] = None):
        self.to_onehot = to_onehot
        self.num_classes = num_classes

    def __call__(self, img, to_onehot: Optional[bool] = None, num_classes: Optional[int] = None):
        """
        Args:
            to_onehot: whether to convert the data to One-Hot format first.
                Defaults to ``self.to_onehot``.
            num_classes: the class number used to convert to One-Hot format if `to_onehot` is True.
                Defaults to ``self.num_classes``.
        """
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
        sigmoid: whether to execute sigmoid function on model output before transform.
            Defaults to ``False``.
        softmax: whether to execute softmax function on model output before transform.
            Defaults to ``False``.
        other: callable function to execute other activation layers, for example:
            `other = lambda x: torch.tanh(x)`. Defaults to ``None``.

    """

    def __init__(self, sigmoid: bool = False, softmax: bool = False, other: Optional[Callable] = None):
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.other = other

    def __call__(
        self, img, sigmoid: Optional[bool] = None, softmax: Optional[bool] = None, other: Optional[Callable] = None
    ):
        """
        Args:
            sigmoid: whether to execute sigmoid function on model output before transform.
                Defaults to ``self.sigmoid``.
            softmax: whether to execute softmax function on model output before transform.
                Defaults to ``self.softmax``.
            other: callable function to execute other activation layers, for example:
                `other = lambda x: torch.tanh(x)`. Defaults to ``self.other``.

        Raises:
            ValueError: sigmoid=True and softmax=True are not compatible.
            ValueError: act_func must be a Callable function.

        """
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
    """
    Execute after model forward to transform model output to discrete values.
    It can complete below operations:

        -  execute `argmax` for input logits values.
        -  threshold input value to 0.0 or 1.0.
        -  convert input value to One-Hot format

    Args:
        argmax: whether to execute argmax function on input data before transform.
            Defaults to ``False``.
        to_onehot: whether to convert input data into the one-hot format.
            Defaults to ``False``.
        n_classes: the number of classes to convert to One-Hot format.
            Defaults to ``None``.
        threshold_values: whether threshold the float value to int number 0 or 1.
            Defaults to ``False``.
        logit_thresh: the threshold value for thresholding operation..
            Defaults to ``0.5``.

    """

    def __init__(
        self,
        argmax: bool = False,
        to_onehot: bool = False,
        n_classes: Optional[int] = None,
        threshold_values: bool = False,
        logit_thresh: float = 0.5,
    ):
        self.argmax = argmax
        self.to_onehot = to_onehot
        self.n_classes = n_classes
        self.threshold_values = threshold_values
        self.logit_thresh = logit_thresh

    def __call__(
        self,
        img,
        argmax: Optional[bool] = None,
        to_onehot: Optional[bool] = None,
        n_classes: Optional[int] = None,
        threshold_values: Optional[bool] = None,
        logit_thresh: Optional[float] = None,
    ):
        """
        Args:
            argmax: whether to execute argmax function on input data before transform.
                Defaults to ``self.argmax``.
            to_onehot: whether to convert input data into the one-hot format.
                Defaults to ``self.to_onehot``.
            n_classes: the number of classes to convert to One-Hot format.
                Defaults to ``self.n_classes``.
            threshold_values: whether threshold the float value to int number 0 or 1.
                Defaults to ``self.threshold_values``.
            logit_thresh: the threshold value for thresholding operation..
                Defaults to ``self.logit_thresh``.

        """
        if argmax or self.argmax:
            img = torch.argmax(img, dim=1, keepdim=True)

        if to_onehot or self.to_onehot:
            _nclasses = self.n_classes if n_classes is None else n_classes
            assert isinstance(_nclasses, int), "One of self.n_classes or n_classes must be an integer"
            img = one_hot(img, _nclasses)

        if threshold_values or self.threshold_values:
            img = img >= (self.logit_thresh if logit_thresh is None else logit_thresh)

        return img.float()


class KeepLargestConnectedComponent(Transform):
    """
    Keeps only the largest connected component in the image.
    This transform can be used as a post-processing step to clean up over-segment areas in model output.

    The input is assumed to be a PyTorch Tensor:
      1) With shape (batch_size, 1, spatial_dim1[, spatial_dim2, ...]) and the values correspond to expected labels.
      2) With shape (batch_size, C, spatial_dim1[, spatial_dim2, ...]) and the values should be 0, 1 on each labels.

    Note:
        For single channel data, 0 will be treated as background and the over-segment pixels will be set to 0.
        For one-hot data, the over-segment pixels will be set to 0 in its channel.

    For example:
    Use KeepLargestConnectedComponent with applied_labels=[1], connectivity=1::

       [1, 0, 0]         [0, 0, 0]
       [0, 1, 1]    =>   [0, 1 ,1]
       [0, 1, 1]         [0, 1, 1]

    Use KeepLargestConnectedComponent with applied_labels[1, 2], independent=False, connectivity=1::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [1, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [1, 2, 0, 0 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,0]

    Use KeepLargestConnectedComponent with applied_labels[1, 2], independent=True, connectivity=1::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [0, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [0, 2, 0, 0 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,0]

    Use KeepLargestConnectedComponent with applied_labels[1, 2], independent=False, connectivity=2::

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [1, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [1, 2, 0, 1 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,2]

    """

    def __init__(self, applied_labels, independent: bool = True, connectivity: Optional[int] = None):
        """
        Args:
            applied_labels (int, list or tuple of int): Labels for applying the connected component on.
                If only one channel. The pixel whose value is not in this list will remain unchanged.
                If the data is in one-hot format, this is used to determine what channels to apply.
            independent (bool): consider several labels as a whole or independent, default is `True`.
                Example use case would be segment label 1 is liver and label 2 is liver tumor, in that case
                you want this "independent" to be specified as False.
            connectivity: Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used.
        """
        super().__init__()
        self.applied_labels = ensure_tuple(applied_labels)
        self.independent = independent
        self.connectivity = connectivity

    def __call__(self, img):
        """
        Args:
            img: shape must be (batch_size, C, spatial_dim1[, spatial_dim2, ...]).

        Returns:
            A PyTorch Tensor with shape (batch_size, C, spatial_dim1[, spatial_dim2, ...]).
        """
        channel_dim = 1
        if img.shape[channel_dim] == 1:

            img = torch.squeeze(img, dim=channel_dim)

            if self.independent:
                for i in self.applied_labels:
                    foreground = (img == i).type(torch.uint8)
                    mask = get_largest_connected_component_mask(foreground, self.connectivity)
                    img[foreground != mask] = 0
            else:
                foreground = torch.zeros_like(img)
                for i in self.applied_labels:
                    foreground += (img == i).type(torch.uint8)
                mask = get_largest_connected_component_mask(foreground, self.connectivity)
                img[foreground != mask] = 0
            output = torch.unsqueeze(img, dim=channel_dim)
        else:
            # one-hot data is assumed to have binary value in each channel
            if self.independent:
                for i in self.applied_labels:
                    foreground = img[:, i, ...].type(torch.uint8)
                    mask = get_largest_connected_component_mask(foreground, self.connectivity)
                    img[:, i, ...][foreground != mask] = 0
            else:
                applied_img = img[:, self.applied_labels, ...].type(torch.uint8)
                foreground = torch.any(applied_img, dim=channel_dim)
                mask = get_largest_connected_component_mask(foreground, self.connectivity)
                background_mask = torch.unsqueeze(foreground != mask, dim=channel_dim)
                background_mask = torch.repeat_interleave(background_mask, len(self.applied_labels), dim=channel_dim)
                applied_img[background_mask] = 0
                img[:, self.applied_labels, ...] = applied_img.type(img.type())
            output = img

        return output


class LabelToContour(Transform):
    """
    Return the contour of binary input images that only compose of 0 and 1, with Laplace kernel
    set as default for edge detection. Typical usage is to plot the edge of label or segmentation output.

    Args:
        kernel_type: the method applied to do edge detection, default is "Laplace".

    """

    def __init__(self, kernel_type: str = "Laplace"):
        if kernel_type != "Laplace":
            raise NotImplementedError("currently, LabelToContour only supports Laplace kernel.")
        self.kernel_type = kernel_type

    def __call__(self, img):
        """
        Args:
            img: torch tensor data to extract the contour, with shape: [batch_size, channels, height, width[, depth]]

        Returns:
            A torch tensor with the same shape as img, note:
                1. it's the binary classification result of whether a pixel is edge or not.
                2. in order to keep the original shape of mask image, we use padding as default.
                3. the edge detection is just approximate because it defects inherent to Laplace kernel,
                   ideally the edge should be thin enough, but now it has a thickness.

        """
        channels = img.shape[1]
        if img.ndim == 4:
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=img.device)
            kernel = kernel.repeat(channels, 1, 1, 1)
            contour_img = F.conv2d(img, kernel, bias=None, stride=1, padding=1, dilation=1, groups=channels)
        elif img.ndim == 5:
            kernel = -1 * torch.ones(3, 3, 3, dtype=torch.float32, device=img.device)
            kernel[1, 1, 1] = 26
            kernel = kernel.repeat(channels, 1, 1, 1, 1)
            contour_img = F.conv3d(img, kernel, bias=None, stride=1, padding=1, dilation=1, groups=channels)
        else:
            raise RuntimeError("the dimensions of img should be 4 or 5.")

        torch.clamp_(contour_img, min=0.0, max=1.0)
        return contour_img
