import torch.nn as nn
import numpy as np

from monai.utils.convutils import samePadding
from monai.networks.utils import getConvType, getDropoutType, getNormalizeType


class Convolution(nn.Sequential):
    def __init__(self, dimensions, inChannels, outChannels, strides=1, kernelSize=3, instanceNorm=True, 
                 dropout=0, dilation=1, bias=True, convOnly=False, isTransposed=False):
        super().__init__()
        self.dimensions = dimensions
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.isTransposed = isTransposed

        padding = samePadding(kernelSize, dilation)
        normalizeType = getNormalizeType(dimensions, instanceNorm)
        convType = getConvType(dimensions, isTransposed)
        dropType = getDropoutType(dimensions)

        if isTransposed:
            conv = convType(inChannels, outChannels, kernelSize, strides, padding, strides - 1, 1, bias, dilation)
        else:
            conv = convType(inChannels, outChannels, kernelSize, strides, padding, dilation, bias=bias)

        self.add_module("conv", conv)

        if not convOnly:
            self.add_module("norm", normalizeType(outChannels))
            if dropout > 0:  # omitting Dropout2d appears faster than relying on it short-circuiting when dropout==0
                self.add_module("dropout", dropType(dropout))

            self.add_module("prelu", nn.modules.PReLU())


class ResidualUnit(nn.Module):
    def __init__(self, dimensions, inChannels, outChannels, strides=1, kernelSize=3, subunits=2, instanceNorm=True, 
                 dropout=0, dilation=1, bias=True, lastConvOnly=False):
        super().__init__()
        self.dimensions = dimensions
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()

        padding = samePadding(kernelSize, dilation)
        schannels = inChannels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            convOnly = lastConvOnly and su == (subunits - 1)
            unit = Convolution(dimensions, schannels, outChannels, sstrides, kernelSize, instanceNorm, dropout, 
                               dilation, bias, convOnly)
            self.conv.add_module("unit%i" % su, unit)
            schannels = outChannels  # after first loop set channels and strides to what they should be for subsequent units
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or inChannels != outChannels:
            rkernelSize = kernelSize
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernelSize = 1
                rpadding = 0

            convType = getConvType(dimensions, False)
            self.residual = convType(inChannels, outChannels, rkernelSize, strides, rpadding, bias=bias)

    def forward(self, x):
        res = self.residual(x)  # create the additive residual from x
        cx = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output
