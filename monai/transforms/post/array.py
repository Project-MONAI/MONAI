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
from monai.transforms.utils import get_largest_connected_component_mask


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


class KeepLargestConnectedComponent(Transform):
    """
    Keeps only the largest connected component in the image.
    This transform can be used as a post-processing step to clean up over-segment areas in model output.
    The input is assumed to be a PyTorch Tensor with shape (batch_size, 1, spatial_dim1[, spatial_dim2, ...])

    Expected input data should have only 1 channel and the values correspond to expected labels.

    For example:
    Use KeepLargestConnectedComponent with applied_values=[1], connectivity=1

       [1, 0, 0]         [0, 0, 0]
       [0, 1, 1]    =>   [0, 1 ,1]
       [0, 1, 1]         [0, 1, 1]

    Use KeepLargestConnectedComponent with applied_values[1, 2], independent=False, connectivity=1

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [1, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [1, 2, 0, 0 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,0]

    Use KeepLargestConnectedComponent with applied_values[1, 2], independent=True, connectivity=1

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [0, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [0, 2, 0, 0 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,0]

    Use KeepLargestConnectedComponent with applied_values[1, 2], independent=False, connectivity=2

      [0, 0, 1, 0 ,0]           [0, 0, 1, 0 ,0]
      [0, 2, 1, 1 ,1]           [0, 2, 1, 1 ,1]
      [1, 2, 1, 0 ,0]    =>     [1, 2, 1, 0 ,0]
      [1, 2, 0, 1 ,0]           [1, 2, 0, 1 ,0]
      [2, 2, 0, 0 ,2]           [2, 2, 0, 0 ,2]

    """

    def __init__(self, applied_values, independent=True, background=0, connectivity=None):
        """
        Args:
            applied_values (list or tuple of int): number list for applying the connected component on.
                The pixel whose value is not in this list will remain unchanged.
            independent (bool): consider several labels as a whole or independent, default is `True`.
                Example use case would be segment label 1 is liver and label 2 is liver tumor, in that case
                you want this "independent" to be specified as False.
            background (int): Background pixel value. The over-segmented pixels will be set as this value.
            connectivity (int): Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
                Accepted values are ranging from  1 to input.ndim. If ``None``, a full
                connectivity of ``input.ndim`` is used.
        """
        super().__init__()
        self.applied_values = applied_values
        self.independent = independent
        self.background = background
        self.connectivity = connectivity
        if background in applied_values:
            raise ValueError("Background pixel can't be in applied_values.")

    def __call__(self, img):
        """
        Args:
            img: shape must be (batch_size, 1, spatial_dim1[, spatial_dim2, ...]).

        Returns:
            A PyTorch Tensor with shape (batch_size, 1, spatial_dim1[, spatial_dim2, ...]).
        """
        channel_dim = 1
        if img.shape[channel_dim] == 1:
            img = torch.squeeze(img, dim=channel_dim)
        else:
            raise ValueError("Input data have more than 1 channel.")

        if self.independent:
            for i in self.applied_values:
                foreground = (img == i).type(torch.uint8)
                mask = get_largest_connected_component_mask(foreground, self.connectivity)
                img[foreground != mask] = self.background
        else:
            foreground = torch.zeros_like(img)
            for i in self.applied_values:
                foreground += (img == i).type(torch.uint8)
            mask = get_largest_connected_component_mask(foreground, self.connectivity)
            img[foreground != mask] = self.background

        return torch.unsqueeze(img, dim=channel_dim)
