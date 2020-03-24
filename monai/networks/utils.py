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
Utilities and types for defining networks, these depend on PyTorch.
"""

import torch
import torch.nn.functional as f


def one_hot(labels, num_classes):
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    num_dims = labels.dim()
    if num_dims < 2 or labels.shape[1] != 1:
        raise ValueError('labels should have a channel with length equals to one.')

    labels = torch.squeeze(labels, 1)
    labels = f.one_hot(labels.long(), num_classes)
    new_axes = [0, -1] + list(range(1, num_dims - 1))
    labels = labels.permute(*new_axes)
    if not labels.is_contiguous():
        return labels.contiguous()
    return labels


def slice_channels(tensor, *slicevals):
    slices = [slice(None)] * len(tensor.shape)
    slices[1] = slice(*slicevals)

    return tensor[slices]


def predict_segmentation(logits, mutually_exclusive=False, threshold=0):
    """
    Given the logits from a network, computing the segmentation by thresholding all values above 0
    if multi-labels task, computing the `argmax` along the channel axis if multi-classes task,
    logits has shape `BCHW[D]`.

    Args:
        logits (Tensor): raw data of model output.
        mutually_exclusive (bool): if True, `logits` will be converted into a binary matrix using
            a combination of argmax, which is suitable for multi-classes task. Defaults to False.
        threshold (float): thresholding the prediction values if multi-labels task.
    """
    if not mutually_exclusive:
        return (logits >= threshold).int()
    else:
        if logits.shape[1] == 1:
            warnings.warn('single channel prediction, `mutually_exclusive=True` ignored, use threshold instead.')
            return (logits >= threshold).int()
        return logits.argmax(1, keepdim=True)
