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

import warnings
import torch
import torch.nn as nn


def one_hot(labels, num_classes: int, dtype: torch.dtype = torch.float):
    """
    For a tensor `labels` of dimensions B1[spatial_dims], return a tensor of dimensions `BN[spatial_dims]`
    for `num_classes` N number of classes.

    Example:

        For every value v = labels[b,1,h,w], the value in the result at [b,v,h,w] will be 1 and all others 0.
        Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    assert labels.dim() > 0, "labels should have dim of 1 or more."

    # if 1D, add singelton dim at the end
    if labels.dim() == 1:
        labels = labels.view(-1, 1)

    sh = list(labels.shape)

    assert sh[1] == 1, "labels should have a channel with length equals to one."
    sh[1] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=1, index=labels.long(), value=1)

    return labels


def slice_channels(tensor: torch.Tensor, *slicevals):
    slices = [slice(None)] * len(tensor.shape)
    slices[1] = slice(*slicevals)

    return tensor[slices]


def predict_segmentation(logits: torch.Tensor, mutually_exclusive: bool = False, threshold: float = 0.0):
    """
    Given the logits from a network, computing the segmentation by thresholding all values above 0
    if multi-labels task, computing the `argmax` along the channel axis if multi-classes task,
    logits has shape `BCHW[D]`.

    Args:
        logits (Tensor): raw data of model output.
        mutually_exclusive: if True, `logits` will be converted into a binary matrix using
            a combination of argmax, which is suitable for multi-classes task. Defaults to False.
        threshold: thresholding the prediction values if multi-labels task.
    """
    if not mutually_exclusive:
        return (logits >= threshold).int()
    else:
        if logits.shape[1] == 1:
            warnings.warn("single channel prediction, `mutually_exclusive=True` ignored, use threshold instead.")
            return (logits >= threshold).int()
        return logits.argmax(1, keepdim=True)


def normalize_transform(shape, device=None, dtype=None, align_corners: bool = False):
    """
    Compute an affine matrix according to the input shape.
    The transform normalizes the homogeneous image coordinates to the
    range of `[-1, 1]`.

    Args:
        shape (sequence of int): input spatial shape
        device (torch device): device on which the returned affine will be allocated.
        dtype (torch dtype): data type of the returned affine
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample
    """
    norm = torch.tensor(shape, dtype=torch.float64, device=device)  # no in-place change
    if align_corners:
        norm[norm <= 1.0] = 2.0
        norm = 2.0 / (norm - 1.0)
        norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64, device=device))))
        norm[:-1, -1] = -1.0
    else:
        norm[norm <= 0.0] = 2.0
        norm = 2.0 / norm
        norm = torch.diag(torch.cat((norm, torch.ones((1,), dtype=torch.float64, device=device))))
        norm[:-1, -1] = 1.0 / torch.tensor(shape, dtype=torch.float64, device=device) - 1.0
    norm = norm.unsqueeze(0).to(dtype=dtype)
    norm.requires_grad = False
    return norm


def to_norm_affine(affine, src_size, dst_size, align_corners: bool = False):
    """
    Given ``affine`` defined for coordinates in the pixel space, compute the corresponding affine
    for the normalized coordinates.

    Args:
        affine (torch Tensor): Nxdxd batched square matrix
        src_size (sequence of int): source image spatial shape
        dst_size (sequence of int): target image spatial shape
        align_corners: if True, consider -1 and 1 to refer to the centers of the
            corner pixels rather than the image corners.
            See also: https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.grid_sample

    Raises:
        ValueError: affine must be a tensor
        ValueError: affine must be Nxdxd, got {tuple(affine.shape)}
        ValueError: affine suggests a {sr}-D transform, but the sizes are src_size={src_size}, dst_size={dst_size}

    """
    if not torch.is_tensor(affine):
        raise ValueError("affine must be a tensor")
    if affine.ndim != 3 or affine.shape[1] != affine.shape[2]:
        raise ValueError(f"affine must be Nxdxd, got {tuple(affine.shape)}")
    sr = affine.shape[1] - 1
    if sr != len(src_size) or sr != len(dst_size):
        raise ValueError(
            f"affine suggests a {sr}-D transform, but the sizes are src_size={src_size}, dst_size={dst_size}"
        )

    src_xform = normalize_transform(src_size, affine.device, affine.dtype, align_corners)
    dst_xform = normalize_transform(dst_size, affine.device, affine.dtype, align_corners)
    new_affine = src_xform @ affine @ torch.inverse(dst_xform)
    return new_affine


def normal_init(m, std=0.02, normal_func=torch.nn.init.normal_):
    """
    Initialize the weight and bias tensors of `m' and its submodules to values from a normal distribution with a
    stddev of `std'. Weight tensors of convolution and linear modules are initialized with a mean of 0, batch
    norm modules with a mean of 1. The callable `normal_func', used to assign values, should have the same arguments
    as its default normal_(). This can be used with `nn.Module.apply` to visit submodules of a network.
    """
    cname = m.__class__.__name__

    if getattr(m, "weight", None) is not None and (cname.find("Conv") != -1 or cname.find("Linear") != -1):
        normal_func(m.weight.data, 0.0, std)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif cname.find("BatchNorm") != -1:
        normal_func(m.weight.data, 1.0, std)
        nn.init.constant_(m.bias.data, 0)
