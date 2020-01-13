import torch.nn as nn

from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.layers.convolutions import Convolution, ResidualUnit
from monai.networks.utils import predictSegmentation
from monai.utils.aliases import alias
from monai.utils import export


@export("monai.networks.nets")
@alias("Unet", "unet")
class UNet(nn.Module):
    def __init__(self, dimensions, inChannels, numClasses, channels, strides, kernelSize=3, upKernelSize=3, 
                 numResUnits=0, instanceNorm=True, dropout=0):
        super().__init__()
        assert len(channels) == (len(strides) + 1)
        self.dimensions = dimensions
        self.inChannels = inChannels
        self.numClasses = numClasses
        self.channels = channels
        self.strides = strides
        self.kernelSize = kernelSize
        self.upKernelSize = upKernelSize
        self.numResUnits = numResUnits
        self.instanceNorm = instanceNorm
        self.dropout = dropout

        def _createBlock(inc, outc, channels, strides, isTop):
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.
            """
            c = channels[0]
            s = strides[0]

            if len(channels) > 2:
                subblock = _createBlock(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._getBottomLayer(c, channels[1])
                upc = c + channels[1]

            down = self._getDownLayer(inc, c, s, isTop)  # create layer in downsampling path
            up = self._getUpLayer(upc, outc, s, isTop)  # create layer in upsampling path

            return nn.Sequential(down, SkipConnection(subblock), up)

        self.model = _createBlock(inChannels, numClasses, self.channels, self.strides, True)

    def _getDownLayer(self, inChannels, outChannels, strides, isTop):
        if self.numResUnits > 0:
            return ResidualUnit(self.dimensions, inChannels, outChannels, strides, self.kernelSize, self.numResUnits, 
                                self.instanceNorm, self.dropout)
        else:
            return Convolution(self.dimensions, inChannels, outChannels, strides, self.kernelSize, 
                               self.instanceNorm, self.dropout)

    def _getBottomLayer(self, inChannels, outChannels):
        return self._getDownLayer(inChannels, outChannels, 1, False)

    def _getUpLayer(self, inChannels, outChannels, strides, isTop):
        conv = Convolution(self.dimensions, inChannels, outChannels, strides, self.upKernelSize, self.instanceNorm, 
                           self.dropout, convOnly=isTop and self.numResUnits == 0, isTransposed=True)

        if self.numResUnits > 0:
            ru = ResidualUnit(self.dimensions, outChannels, outChannels, 1, self.kernelSize, 1, self.instanceNorm, 
                              self.dropout, lastConvOnly=isTop)
            return nn.Sequential(conv, ru)
        else:
            return conv

    def forward(self, x):
        x = self.model(x)
        return x, predictSegmentation(x)
