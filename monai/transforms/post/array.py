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
from copy import deepcopy
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

from monai.networks import one_hot
from monai.networks.layers import GaussianFilter
from monai.transforms.transform import NumpyTransform, TorchTransform
from monai.transforms.utils import get_largest_connected_component_mask
from monai.utils import ensure_tuple
from monai.utils.enums import DataObjects
from monai.utils.misc import convert_data_type

__all__ = [
    "Activations",
    "AsDiscrete",
    "KeepLargestConnectedComponent",
    "LabelToContour",
    "MeanEnsemble",
    "VoteEnsemble",
    "ProbNMS",
]


def _sigmoid(z):
    if isinstance(z, torch.Tensor):
        return torch.sigmoid(z)
    return 1 / (1 + np.exp(-z))


def _softmax(z, dim):
    if isinstance(z, torch.Tensor):
        return torch.softmax(z, dim=dim)

    max = np.max(z, axis=dim, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(z - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=dim, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


class Activations(TorchTransform, NumpyTransform):
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
        img: DataObjects.Images,
        sigmoid: Optional[bool] = None,
        softmax: Optional[bool] = None,
        other: Optional[Callable] = None,
    ) -> DataObjects.Images:
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
        if sigmoid or self.sigmoid:
            img = _sigmoid(img)
        if softmax or self.softmax:
            img = _softmax(img, dim=0)

        act_func = self.other if other is None else other
        if act_func is not None:
            try:
                img = act_func(img)
            except TypeError as te:
                # callable only works on torch.Tensors
                if "must be Tensor, not numpy.ndarray" in str(te):
                    img, *_ = convert_data_type(img, torch.Tensor)
                    img = act_func(img)
                    img, *_ = convert_data_type(img, np.ndarray)

        return img


class AsDiscrete(TorchTransform, NumpyTransform):
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
        img: DataObjects.Images,
        argmax: Optional[bool] = None,
        to_onehot: Optional[bool] = None,
        n_classes: Optional[int] = None,
        threshold_values: Optional[bool] = None,
        logit_thresh: Optional[float] = None,
    ) -> DataObjects.Images:
        """
        Args:
            img: the input tensor data to convert, if no channel dimension when converting to `One-Hot`,
                will automatically add it.
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
            if isinstance(img, torch.Tensor):
                img = torch.argmax(img, dim=0, keepdim=True)
            else:
                img = np.argmax(img, axis=0)[None]

        if to_onehot or self.to_onehot:
            _nclasses = self.n_classes if n_classes is None else n_classes
            if not isinstance(_nclasses, int):
                raise AssertionError("One of self.n_classes or n_classes must be an integer")
            img = one_hot(img, num_classes=_nclasses, dim=0)

        if threshold_values or self.threshold_values:
            img = img >= (logit_thresh or self.logit_thresh)

        out, *_ = convert_data_type(img, dtype=torch.float32)
        return out


class KeepLargestConnectedComponent(TorchTransform):
    """
    Keeps only the largest connected component in the image.
    This transform can be used as a post-processing step to clean up over-segment areas in model output.

    The input is assumed to be a channel-first PyTorch Tensor:
      1) With shape (1, spatial_dim1[, spatial_dim2, ...]) and the values correspond to expected labels.
      2) With shape (C, spatial_dim1[, spatial_dim2, ...]) and the values should be 0, 1 on each labels.

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

    @staticmethod
    def _astype(x, dtype=torch.uint8):
        x, *_ = convert_data_type(x, dtype=dtype)
        return x

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: shape must be (C, spatial_dim1[, spatial_dim2, ...]).

        Returns:
            A PyTorch Tensor with shape (C, spatial_dim1[, spatial_dim2, ...]).
        """
        if img.shape[0] == 1:
            img = img.squeeze(0)

            if self.independent:
                for i in self.applied_labels:
                    foreground = self._astype(img == i)
                    mask = get_largest_connected_component_mask(foreground, self.connectivity)
                    img[foreground != mask] = 0
            else:
                foreground = torch.zeros_like(img) if isinstance(img, torch.Tensor) else np.zeros_like(img)
                for i in self.applied_labels:
                    foreground += self._astype(img == i)
                mask = get_largest_connected_component_mask(foreground, self.connectivity)
                img[foreground != mask] = 0

            output = img[None]
        else:
            # one-hot data is assumed to have binary value in each channel
            if self.independent:
                for i in self.applied_labels:
                    foreground = self._astype(img[i, ...])
                    mask = get_largest_connected_component_mask(foreground, self.connectivity)
                    img[i, ...][foreground != mask] = 0
            else:
                applied_img = self._astype(img[self.applied_labels, ...])
                foreground = applied_img.any(0)
                mask = get_largest_connected_component_mask(foreground, self.connectivity)
                background_mask = (foreground != mask)[None]
                if isinstance(background_mask, torch.Tensor):
                    background_mask = torch.repeat_interleave(background_mask, len(self.applied_labels), dim=0)
                else:
                    background_mask = np.repeat(background_mask, len(self.applied_labels), axis=0)
                applied_img[background_mask] = 0
                img[self.applied_labels, ...] = self._astype(applied_img, img.dtype)
            output = img

        return output


class LabelToContour(TorchTransform):
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

    def __call__(self, img: DataObjects.Images) -> DataObjects.Images:
        """
        Args:
            img: torch tensor data to extract the contour, with shape: [channels, height, width[, depth]]

        Raises:
            ValueError: When ``image`` ndim is not one of [3, 4].

        Returns:
            A torch tensor with the same shape as img, note:
                1. it's the binary classification result of whether a pixel is edge or not.
                2. in order to keep the original shape of mask image, we use padding as default.
                3. the edge detection is just approximate because it defects inherent to Laplace kernel,
                   ideally the edge should be thin enough, but now it has a thickness.

        """
        img_t: torch.Tensor
        img_t, orig_type, orig_device = convert_data_type(img, torch.Tensor)  # type: ignore

        channels = img_t.shape[0]
        img_ = img_t.unsqueeze(0)
        if img_t.ndimension() == 3:
            kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32, device=img_t.device)
            kernel = kernel.repeat(channels, 1, 1, 1)
            contour_img = F.conv2d(img_, kernel, bias=None, stride=1, padding=1, dilation=1, groups=channels)
        elif img_t.ndimension() == 4:
            kernel = -1 * torch.ones(3, 3, 3, dtype=torch.float32, device=img_t.device)
            kernel[1, 1, 1] = 26
            kernel = kernel.repeat(channels, 1, 1, 1, 1)
            contour_img = F.conv3d(img_, kernel, bias=None, stride=1, padding=1, dilation=1, groups=channels)
        else:
            raise ValueError(f"Unsupported img dimension: {img_t.ndimension()}, available options are [4, 5].")

        contour_img.clamp_(min=0.0, max=1.0)
        contour_img = contour_img.squeeze(0)

        out, *_ = convert_data_type(contour_img, orig_type, orig_device)
        return out


class MeanEnsemble(TorchTransform, NumpyTransform):
    """
    Execute mean ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E, C[, H, W, D]], the `E` dimension represents
    the output data from different models.
    Typically, the input data is model output of segmentation task or classification task.
    And it also can support to add `weights` for the input data.

    Args:
        weights: can be a list or tuple of numbers for input data with shape: [E, C, H, W[, D]].
            or a Numpy ndarray or a PyTorch Tensor data.
            the `weights` will be added to input data from highest dimension, for example:
            1. if the `weights` only has 1 dimension, it will be added to the `E` dimension of input data.
            2. if the `weights` has 2 dimensions, it will be added to `E` and `C` dimensions.
            it's a typical practice to add weights for different classes:
            to ensemble 3 segmentation model outputs, every output has 4 channels(classes),
            so the input data shape can be: [3, 4, H, W, D].
            and add different `weights` for different classes, so the `weights` shape can be: [3, 4].
            for example: `weights = [[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 1, 1]]`.

    """

    def __init__(self, weights: Optional[Union[Sequence[float], DataObjects.Images]] = None) -> None:
        if weights is None:
            self.weights = None
        elif isinstance(weights, (torch.Tensor, np.ndarray)):
            self.weights = weights
        else:
            self.weights = torch.as_tensor(weights, dtype=torch.float)

    def __call__(self, img: Union[Sequence[DataObjects.Images], DataObjects.Images]) -> DataObjects.Images:
        if isinstance(img, (torch.Tensor, np.ndarray)):
            img_ = img
        elif isinstance(img[0], torch.Tensor):
            img_ = torch.stack(img)  # type: ignore
        else:
            img_ = np.stack(img)

        if self.weights is not None:
            self.weights, *_ = convert_data_type(
                self.weights, type(img_), device=img_.device if isinstance(img_, torch.Tensor) else None
            )
            shape = tuple(self.weights.shape)
            for _ in range(img_.ndim - self.weights.ndim):
                shape += (1,)
            weights = self.weights.reshape(*shape)

            if isinstance(img_, torch.Tensor):
                # torch can only do the mean on floats
                img_ = img_ * weights / weights.float().mean(dim=0, keepdim=True)  # type: ignore
            else:
                img_ = img_ * weights / weights.mean(axis=0, keepdims=True)  # type: ignore

        return img_.mean(0)  # type: ignore


class VoteEnsemble(TorchTransform, NumpyTransform):
    """
    Execute vote ensemble on the input data.
    The input data can be a list or tuple of PyTorch Tensor with shape: [C[, H, W, D]],
    Or a single PyTorch Tensor with shape: [E[, C, H, W, D]], the `E` dimension represents
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

    def __call__(self, img: Union[Sequence[DataObjects.Images], DataObjects.Images]) -> DataObjects.Images:
        if isinstance(img, (torch.Tensor, np.ndarray)):
            img_ = img
        elif isinstance(img[0], torch.Tensor):
            img_ = torch.stack(img)  # type: ignore
        else:
            img_ = np.stack(img)

        if self.num_classes is not None:
            has_ch_dim = True
            if img_.ndim > 1 and img_.shape[1] > 1:
                warnings.warn("no need to specify num_classes for One-Hot format data.")
            else:
                if img_.ndim == 1:
                    # if no channel dim, need to remove channel dim after voting
                    has_ch_dim = False
                img_ = one_hot(img_, self.num_classes, dim=1)

        img_ = torch.mean(img_.float(), dim=0) if isinstance(img_, torch.Tensor) else np.mean(img_, axis=0)

        if self.num_classes is not None:
            # if not One-Hot, use "argmax" to vote the most common class
            if isinstance(img_, torch.Tensor):
                return torch.argmax(img_, dim=0, keepdim=has_ch_dim)
            else:
                img_ = np.argmax(img_, axis=0)
                img_ = np.array(img_) if np.isscalar(img_) else img_  # numpy returns scalar if input was 1d
                return img_[None] if has_ch_dim else img_
        # for One-Hot data, round the float number to 0 or 1
        return torch.round(img_) if isinstance(img_, torch.Tensor) else np.round(img_)


class ProbNMS(TorchTransform):
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
        prob_map: DataObjects.Images,
    ) -> List[List]:
        """
        prob_map: the input probabilities map, it must have shape (H[, W, ...]).
        """
        prob_map_t: torch.Tensor
        prob_map_t, *_ = convert_data_type(deepcopy(prob_map), torch.Tensor, dtype=torch.float32)  # type: ignore
        if self.sigma != 0:
            self.filter.to(prob_map_t)
            prob_map_t = self.filter(prob_map_t)

        prob_map_shape = prob_map_t.shape

        outputs = []
        while prob_map_t.max() > self.prob_threshold:
            max_idx = np.unravel_index(prob_map_t.argmax().cpu(), prob_map_shape)
            prob_max = prob_map_t[max_idx].item()
            max_idx_arr = np.asarray(max_idx)
            outputs.append([prob_max] + list(max_idx_arr))

            idx_min_range = (max_idx_arr - self.box_lower_bd).clip(0, None)
            idx_max_range = (max_idx_arr + self.box_upper_bd).clip(None, prob_map_shape)
            # for each dimension, set values during index ranges to 0
            slices = tuple(slice(idx_min_range[i], idx_max_range[i]) for i in range(self.spatial_dims))
            prob_map_t[slices] = 0

        return outputs
