"""
Utilities and types for defining networks, these depend on Pytorch.
"""

import torch
import torch.nn as nn


def one_hot(labels, num_classes):
    """
    For a tensor `labels' of dimensions BC[D][H]W, return a tensor of dimensions BC[D][H]WN for `num_classes' N number of
    classes. For every value v = labels[b,c,h,w], the value in the result at [b,c,h,w,v] will be 1 and all others 0. 
    Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    """
    onehotshape = tuple(labels.shape) + (num_classes,)
    labels = labels % num_classes
    y = torch.eye(num_classes, device=labels.device)
    onehot = y[labels.view(-1).long()]

    return onehot.reshape(*onehotshape)


def slice_channels(tensor, *slicevals):
    slices = [slice(None)] * len(tensor.shape)
    slices[1] = slice(*slicevals)

    return tensor[slices]


def predict_segmentation(logits):
    """
    Given the logits from a network, computing the segmentation by thresholding all values above 0 if `logits' has one
    channel, or computing the argmax along the channel axis otherwise.
    """
    # generate prediction outputs, logits has shape BCHW[D]
    if logits.shape[1] == 1:
        return (logits[:, 0] >= 0).int()  # for binary segmentation threshold on channel 0
    else:
        return logits.max(1)[1]  # take the index of the max value along dimension 1


def get_conv_type(dim, is_transpose):
    if is_transpose:
        types = [nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d]
    else:
        types = [nn.Conv1d, nn.Conv2d, nn.Conv3d]

    return types[dim - 1]


def get_dropout_type(dim):
    types = [nn.Dropout, nn.Dropout2d, nn.Dropout3d]
    return types[dim - 1]


def get_normalize_type(dim, is_instance):
    if is_instance:
        types = [nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d]
    else:
        types = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]

    return types[dim - 1]
