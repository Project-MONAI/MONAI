
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
Utilities and types for defining networks, these depend on Pytorch.
"""

import torch
import torch.nn as nn


def oneHot(labels, numClasses):
    """
    For a tensor `labels' of dimensions BC[D][H]W, return a tensor of dimensions BC[D][H]WN for `numClasses' N number of 
    classes. For every value v = labels[b,c,h,w], the value in the result at [b,c,h,w,v] will be 1 and all others 0. 
    Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    onehotshape = tuple(labels.shape) + (numClasses,)
    labels = labels % numClasses
    y = torch.eye(numClasses, device=labels.device)
    onehot = y[labels.view(-1).long()]

    return onehot.reshape(*onehotshape)


def sliceChannels(tensor, *slicevals):
    slices = [slice(None)] * len(tensor.shape)
    slices[1] = slice(*slicevals)

    return tensor[slices]


def predictSegmentation(logits):
    """
    Given the logits from a network, computing the segmentation by thresholding all values above 0 if `logits' has one
    channel, or computing the argmax along the channel axis otherwise.
    """
    # generate prediction outputs, logits has shape BCHW[D]
    if logits.shape[1] == 1:
        return (logits[:, 0] >= 0).int()  # for binary segmentation threshold on channel 0
    else:
        return logits.max(1)[1]  # take the index of the max value along dimension 1


def getConvType(dim, isTranspose):
    if isTranspose:
        types = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    else:
        types = [nn.Conv1d, nn.Conv2d, nn.Conv3d]

    return types[dim - 1]


def getDropoutType(dim):
    types = [nn.Dropout, nn.Dropout2d, nn.Dropout3d]
    return types[dim - 1]


def getNormalizeType(dim, isInstance):
    if isInstance:
        types = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    else:
        types = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

    return types[dim - 1]

