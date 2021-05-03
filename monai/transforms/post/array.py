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
"""
A collection of "vanilla" transforms for the model output tensors
https://github.com/Project-MONAI/MONAI/wiki/MONAI_Design
"""

import warnings
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from monai.networks import one_hot
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import Transform
from monai.transforms.utils import get_largest_connected_component_mask
from monai.utils import ensure_tuple

__all__ = [
    "Activations",
    "AsDiscrete",
    "KeepLargestConnectedComponent",
    "LabelToContour",
    "MeanEnsemble",
    "VoteEnsemble",
    "ProbNMS",
]


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

    Raises:
        TypeError: When ``other`` is not an ``Optional[Callable]``.

    """

    def __init__(self, sigmoid: bool = False, softmax: bool = False, other: Optional[Callable] = None) -> None:
        self.sigmoid = sigmoid
        self.softmax = softmax
        if other is not None and not callable(other):
            raise TypeError(f"other must be None or callable but is {type(other).__name__}.")
        self.other = other

    def __call__(
        self,
        img: torch.Tensor,
        sigmoid: Optional[bool] = None,
        softmax: Optional[bool] = None,
        other: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Args:
            sigmoid: whether to execute sigmoid function on model output before transform.
                Defaults to ``self.sigmoid``.
            softmax: whether to execute softmax function on model output before transform.
                Defaults to ``self.softmax``.
            other: callable function to execute other activation layers, for example:
                `other = torch.tanh`. Defaults to ``self.other``.

        Raises:
            ValueError: When ``sigmoid=True`` and ``softmax=True``. Incompatible values.
            TypeError: When ``other`` is not an ``Optional[Callable]``.
            ValueError: When ``self.other=None`` and ``other=None``. Incompatible values.

        """
        if sigmoid and softmax:
            raise ValueError("Incompatible values: sigmoid=True and softmax=True.")
        if other is not None and not callable(other):
            raise TypeError(f"other must be None or callable but is {type(other).__name__}.")

        # convert to float as activation must operate on float tensor
        img = img.float()
        if sigmoid or self.sigmoid:
            img = torch.sigmoid(img)
        if softmax or self.softmax:
            # add channel dim if not existing
            if img.ndimension() == 1:
                img = img.unsqueeze(-1)
            img = torch.softmax(img, dim=1)

        act_func = self.other if other is None else other
        if act_func is not None:
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
    ) -> None:
        self.argmax = argmax
        self.to_onehot = to_onehot
        self.n_classes = n_classes
        self.threshold_values = threshold_values
        self.logit_thresh = logit_thresh

    def __call__(
        self,
        img: torch.Tensor,
        argmax: Optional[bool] = None,
        to_onehot: Optional[bool] = None,
        n_classes: Optional[int] = None,
        threshold_values: Optional[bool] = None,
        logit_thresh: Optional[float] = None,
    ) -> torch.Tensor:
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
            if not isinstance(_nclasses, int):
                raise AssertionError("One of self.n_classes or n_classes must be an integer")
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

    def __init__(
        self, applied_labels: Union[Sequence[int], int], independent: bool = True, connectivity: Optional[int] = None
    ) -> None:
        """
        Args:
            applied_labels: Labels for applying the connected component on.
                If only one channel. The pixel whose value is not in this list will remain unchanged.
                If the data is in one-hot format, this is used to determine what channels to apply.
            independent: consider several labels as a whole or independent, default is `True`.
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

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
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

    Raises:
        NotImplementedError: When ``kernel_type`` is not "Laplace".

    """

    def __init__(self, kernel_type: str = "Laplace") -> None:
        if kernel_type != "Laplace":
            raise NotImplementedError('Currently only kernel_type="Laplace" is supported.')
        self.kernel_type = kernel_type

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: torch tensor data to extract the contour, with shape: [batch_size, channels, height, width[, depth]]

        Raises:
            ValueError: When ``image`` ndim is not one of [4, 5].

        Returns:
            A torch tensor with the same shape as img, note:
                1. it's the binary classification result of whether a pixel is edge or not.
                2. in order to keep the original shape of mask image, we use padding as default.
                3. the edge detection is just approximate because it defects inherent to Laplace kernel,
                   ideally the edge should be thin enough, but now it has a thickness.

        """
        channels = img.shape[1]
        if img.ndimension() == 4:
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=img.device)
            kernel = kernel.repeat(channels, 1, 1, 1)
            contour_img = F.conv2d(img, kernel, bias=None, stride=1, padding=1, dilation=1, groups=channels)
        elif img.ndimension() == 5:
            kernel = -1 * torch.ones(3, 3, 3, dtype=torch.float32, device=img.device)
            kernel[1, 1, 1] = 26
            kernel = kernel.repeat(channels, 1, 1, 1, 1)
            contour_img = F.conv3d(img, kernel, bias=None, stride=1, padding=1, dilation=1, groups=channels)
        else:
            raise ValueError(f"Unsupported img dimension: {img.ndimension()}, available options are [4, 5].")

        contour_img.clamp_(min=0.0, max=1.0)
        return contour_img


class MeanEnsemble(Transform):
    """
    Execute mean ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [B, C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E, B, C[, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is model output of segmentation task or classification task.
    And it also can support to add `weights` for the input data.

    Args:
        weights: can be a list or tuple of numbers for input data with shape: [E, B, C, H, W[, D]].
            or a Numpy ndarray or a PyTorch Tensor data.
            the `weights` will be added to input data from highest dimension, for example:
            1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
            2. if the `weights` has 3 dimensions, it will be added to `E`, `B` and `C` dimensions.
            it's a typical practice to add weights for different classes:
            to ensemble 3 segmentation model outputs, every output has 4 channels(classes),
            so the input data shape can be: [3, B, 4, H, W, D].
            and add different `weights` for different classes, so the `weights` shape can be: [3, 1, 4].
            for example: `weights = [[[1, 2, 3, 4]], [[4, 3, 2, 1]], [[1, 1, 1, 1]]]`.

    """

    def __init__(self, weights: Optional[Union[Sequence[float], torch.Tensor, np.ndarray]] = None) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.float) if weights is not None else None

    def __call__(self, img: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        img_ = torch.stack(img) if isinstance(img, (tuple, list)) else torch.as_tensor(img)
        if self.weights is not None:
            self.weights = self.weights.to(img_.device)
            shape = tuple(self.weights.shape)
            for _ in range(img_.ndimension() - self.weights.ndimension()):
                shape += (1,)
            weights = self.weights.reshape(*shape)

            img_ = img_ * weights / weights.mean(dim=0, keepdim=True)

        return torch.mean(img_, dim=0)


class VoteEnsemble(Transform):
    """
    Execute vote ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [B[, C, H, W, D]],
    Or a single PyTorch Tensor with shape: [E, B[, C, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is model output of segmentation task or classification task.

    Note:
        This vote transform expects the input data is discrete values. It can be multiple channels
        data in One-Hot format or single channel data. It will vote to select the most common data
        between items.
        The output data has the same shape as every item of the input data.

    Args:
        num_classes: if the input is single channel data instead of One-Hot, we can't get class number
            from channel, need to explicitly specify the number of classes to vote.

    """

    def __init__(self, num_classes: Optional[int] = None) -> None:
        self.num_classes = num_classes

    def __call__(self, img: Union[Sequence[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        img_ = torch.stack(img) if isinstance(img, (tuple, list)) else torch.as_tensor(img)
        if self.num_classes is not None:
            has_ch_dim = True
            if img_.ndimension() > 2 and img_.shape[2] > 1:
                warnings.warn("no need to specify num_classes for One-Hot format data.")
            else:
                if img_.ndimension() == 2:
                    # if no channel dim, need to remove channel dim after voting
                    has_ch_dim = False
                img_ = one_hot(img_, self.num_classes, dim=2)

        img_ = torch.mean(img_.float(), dim=0)

        if self.num_classes is not None:
            # if not One-Hot, use "argmax" to vote the most common class
            return torch.argmax(img_, dim=1, keepdim=has_ch_dim)
        # for One-Hot data, round the float number to 0 or 1
        return torch.round(img_)


class ProbNMS(Transform):
    """
    Performs probability based non-maximum suppression (NMS) on the probabilities map via
    iteratively selecting the coordinate with highest probability and then move it as well
    as its surrounding values. The remove range is determined by the parameter `box_size`.
    If multiple coordinates have the same highest probability, only one of them will be
    selected.

    Args:
        spatial_dims: number of spatial dimensions of the input probabilities map.
            Defaults to 2.
        sigma: the standard deviation for gaussian filter.
            It could be a single value, or `spatial_dims` number of values. Defaults to 0.0.
        prob_threshold: the probability threshold, the function will stop searching if
            the highest probability is no larger than the threshold. The value should be
            no less than 0.0. Defaults to 0.5.
        box_size: the box size (in pixel) to be removed around the the pixel with the maximum probability.
            It can be an integer that defines the size of a square or cube,
            or a list containing different values for each dimensions. Defaults to 48.

    Return:
        a list of selected lists, where inner lists contain probability and coordinates.
        For example, for 3D input, the inner lists are in the form of [probability, x, y, z].

    Raises:
        ValueError: When ``prob_threshold`` is less than 0.0.
        ValueError: When ``box_size`` is a list or tuple, and its length is not equal to `spatial_dims`.
        ValueError: When ``box_size`` has a less than 1 value.

    """

    def __init__(
        self,
        spatial_dims: int = 2,
        sigma: Union[Sequence[float], float, Sequence[torch.Tensor], torch.Tensor] = 0.0,
        prob_threshold: float = 0.5,
        box_size: Union[int, Sequence[int]] = 48,
    ) -> None:
        self.sigma = sigma
        self.spatial_dims = spatial_dims
        if self.sigma != 0:
            self.filter = GaussianFilter(spatial_dims=spatial_dims, sigma=sigma)
        if prob_threshold < 0:
            raise ValueError("prob_threshold should be no less than 0.0.")
        self.prob_threshold = prob_threshold
        if isinstance(box_size, int):
            self.box_size = np.asarray([box_size] * spatial_dims)
        else:
            if len(box_size) != spatial_dims:
                raise ValueError("the sequence length of box_size should be the same as spatial_dims.")
            self.box_size = np.asarray(box_size)
        if self.box_size.min() <= 0:
            raise ValueError("box_size should be larger than 0.")

        self.box_lower_bd = self.box_size // 2
        self.box_upper_bd = self.box_size - self.box_lower_bd

    def __call__(
        self,
        prob_map: Union[np.ndarray, torch.Tensor],
    ):
        """
        prob_map: the input probabilities map, it must have shape (H[, W, ...]).
        """
        if self.sigma != 0:
            if not isinstance(prob_map, torch.Tensor):
                prob_map = torch.as_tensor(prob_map, dtype=torch.float)
            self.filter.to(prob_map)
            prob_map = self.filter(prob_map)
        else:
            if not isinstance(prob_map, torch.Tensor):
                prob_map = prob_map.copy()

        if isinstance(prob_map, torch.Tensor):
            prob_map = prob_map.detach().cpu().numpy()

        prob_map_shape = prob_map.shape

        outputs = []
        while np.max(prob_map) > self.prob_threshold:
            max_idx = np.unravel_index(prob_map.argmax(), prob_map_shape)
            prob_max = prob_map[max_idx]
            max_idx_arr = np.asarray(max_idx)
            outputs.append([prob_max] + list(max_idx_arr))

            idx_min_range = (max_idx_arr - self.box_lower_bd).clip(0, None)
            idx_max_range = (max_idx_arr + self.box_upper_bd).clip(None, prob_map_shape)
            # for each dimension, set values during index ranges to 0
            slices = tuple(slice(idx_min_range[i], idx_max_range[i]) for i in range(self.spatial_dims))
            prob_map[slices] = 0

        return outputs
