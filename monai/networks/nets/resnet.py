import math
from functools import partial
from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Dropout, Norm, Pool

__all__ = ["ResNet", "resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200"]

def get_inplanes():
    return [64, 128, 256, 512]

def get_avgpool():
    return [(0), (1), (1,1), (1,1,1)]

class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self, 
        in_planes: int, 
        planes: int, 
        spatial_dims: int = 3,
        stride: int = 1, 
        downsample: Optional[nn.Module] = None
        ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: if to use the downsample_basic_block
        """
        super(ResNetBlock, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.add_module("conv1", conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False))
        self.add_module("bn1", norm_type(planes))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv2", conv_type(planes, planes, kernel_size=3, padding=1, bias=False))
        self.add_module("bn2", norm_type(planes))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self, 
        in_planes:int,
        planes:int,
        spatial_dims: int = 3,
        stride:int = 1,
        downsample: Optional[nn.Module] = None
        ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: if to use the downsample_basic_block
        """
        
        super(ResNetBottleneck, self).__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.add_module("conv1", conv_type(in_planes, planes, kernel_size=1, padding=1, bias=False))
        self.add_module("bn1", norm_type(planes))
        self.add_module("conv2", conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False))
        self.add_module("bn2", norm_type(planes))
        self.add_module("conv3", conv_type(planes, planes * self.expansion, kernel_size=1, padding=1, bias=False))
        self.add_module("bn3", norm_type(planes * self.expansion))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs <https://arxiv.org/pdf/2004.04968.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.
    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tuneable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use.
        widen_factor: widen output for each layer.
        n_classes: number of output (classifications) 
    """
    def __init__(
        self,
        block:Type[Union[ResNetBlock, ResNetBottleneck]],
        layers:List[int],
        block_inplanes:List[int],
        block_avgpool:List[int],
        spatial_dims:int = 3,
        n_input_channels:int = 3,
        conv1_t_size:int = 7,
        conv1_t_stride: int = 1,
        no_max_pool:bool = False,
        shortcut_type:str = 'B',
        widen_factor:float = 1.0,
        n_classes:int = 400
        ) -> None:

        super(ResNet, self).__init__()

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avgp_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[Pool.ADAPTIVEAVG, spatial_dims]

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        if spatial_dims == 1:
            self.add_module("conv1", conv_type(n_input_channels, 
                                            self.in_planes, 
                                            kernel_size=(conv1_t_size), 
                                            stride=(conv1_t_stride), 
                                            padding=(conv1_t_size // 2), 
                                            bias=False))
        elif spatial_dims == 2:
            self.add_module("conv1", conv_type(n_input_channels, 
                                            self.in_planes, 
                                            kernel_size=(conv1_t_size, 7), 
                                            stride=(conv1_t_stride, 2), 
                                            padding=(conv1_t_size // 2, 3), 
                                            bias=False))
        else:
            self.add_module("conv1", conv_type(n_input_channels, 
                                            self.in_planes, 
                                            kernel_size=(conv1_t_size, 7, 7), 
                                            stride=(conv1_t_stride, 2, 2), 
                                            padding=(conv1_t_size // 2, 3, 3), 
                                            bias=False))

        self.add_module("bn1", norm_type(self.in_planes))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("maxpool", pool_type(kernel_size=3, stride=2, padding=1))
        self.add_module("layer1", self._make_layer(block, 
                                                   block_inplanes[0],
                                                   layers[0],
                                                   spatial_dims,
                                                   shortcut_type))
        self.add_module("layer2", self._make_layer(block, 
                                                   block_inplanes[1],
                                                   layers[1],
                                                   spatial_dims,
                                                   shortcut_type,
                                                   stride=2))
        self.add_module("layer3", self._make_layer(block, 
                                                   block_inplanes[2],
                                                   layers[2],
                                                   spatial_dims,
                                                   shortcut_type,
                                                   stride=2))
        self.add_module("layer4", self._make_layer(block, 
                                                   block_inplanes[3],
                                                   layers[3],
                                                   spatial_dims,
                                                   shortcut_type,
                                                   stride=2))

        self.add_module("avgpool", avgp_type(block_avgpool[spatial_dims]))
        
        self.add_module("fc", nn.Linear(block_inplanes[3] * block.expansion, n_classes))

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, norm_type):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes:int, stride:int, spatial_dims: int = 3) -> torch.Tensor:
        assert spatial_dims == 3
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(
        self, 
        block: Type[Union[ResNetBlock, ResNetBottleneck]],
        planes:int,
        blocks:int, 
        spatial_dims:int, 
        shortcut_type:str, 
        stride:int = 1
        ) -> nn.Sequential:

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     kernel_size=1,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv_type(self.in_planes, 
                              planes * block.expansion,
                              kernel_size=1,
                              stride=stride),
                    norm_type(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  spatial_dims=spatial_dims,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet10(**kwargs):
    """ResNet-10 with optional pretrained support when `spatial_dims` is 3."""
    model = ResNet(ResNetBlock, [1, 1, 1, 1], get_inplanes(), get_avgpool(), **kwargs)
    return model


def resnet18(**kwargs):
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3."""
    model = ResNet(ResNetBlock, [2, 2, 2, 2], get_inplanes(), get_avgpool(), **kwargs)
    return model


def resnet34(**kwargs):
    """ResNet-34 with optional pretrained support when `spatial_dims` is 3."""
    model = ResNet(ResNetBlock, [3, 4, 6, 3], get_inplanes(), get_avgpool(), **kwargs)
    return model


def resnet50(**kwargs):
    """ResNet-50 with optional pretrained support when `spatial_dims` is 3."""
    model = ResNet(ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), get_avgpool(), **kwargs)
    return model


def resnet101(**kwargs):
    """ResNet-101 with optional pretrained support when `spatial_dims` is 3."""
    model = ResNet(ResNetBottleneck, [3, 4, 23, 3], get_inplanes(), get_avgpool(), **kwargs)
    return model


def resnet152(**kwargs):
    """ResNet-152 with optional pretrained support when `spatial_dims` is 3."""
    model = ResNet(ResNetBottleneck, [3, 8, 36, 3], get_inplanes(), get_avgpool(), **kwargs)
    return model


def resnet200(**kwargs):
    """ResNet-200 with optional pretrained support when `spatial_dims` is 3."""
    model = ResNet(ResNetBottleneck, [3, 24, 36, 3], get_inplanes(), get_avgpool(), **kwargs)
    return model

if __name__=="__main__":
    print(resnet101(n_input_channels=1, n_classes=2, spatial_dims=2, ))
    #resnet152(n_input_channels=1, n_classes=2, spatial_dims=3, )