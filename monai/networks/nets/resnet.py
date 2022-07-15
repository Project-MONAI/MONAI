# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option

__all__ = ["ResNet", "resnet10", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnet200"]

from monai.utils import deprecated_arg


def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
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
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: Union[nn.Module, partial, None] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
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
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.

    .. deprecated:: 0.6.0
        ``n_classes`` is deprecated, use ``num_classes`` instead.

    """

    @deprecated_arg("n_classes", since="0.6")
    def __init__(
        self,
        block: Union[Type[Union[ResNetBlock, ResNetBottleneck]], str],
        layers: List[int],
        block_inplanes: List[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: Union[Tuple[int], int] = 7,
        conv1_t_stride: Union[Tuple[int], int] = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        n_classes: Optional[int] = None,
    ) -> None:

        super().__init__()
        # in case the new num_classes is default but you still call deprecated n_classes
        if n_classes is not None and num_classes == 400:
            num_classes = n_classes

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        norm_type: Type[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avgp_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: Type[Union[ResNetBlock, ResNetBottleneck]],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: Union[nn.Module, partial, None] = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride),
                    norm_type(planes * block.expansion),
                )

        layers = [
            block(
                in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims, stride=stride, downsample=downsample
            )
        ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
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
        if self.fc is not None:
            x = self.fc(x)

        return x


def _resnet(
    arch: str,
    block: Type[Union[ResNetBlock, ResNetBottleneck]],
    layers: List[int],
    block_inplanes: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model: ResNet = ResNet(block, layers, block_inplanes, **kwargs)
    if pretrained:
        # Author of paper zipped the state_dict on googledrive,
        # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
        # Would like to load dict from url but need somewhere to save the state dicts.
        raise NotImplementedError(
            "Currently not implemented. You need to manually download weights provided by the paper's author"
            " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
        )
    return model


def resnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-10 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes(), pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", ResNetBottleneck, [3, 4, 23, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", ResNetBottleneck, [3, 8, 36, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet200(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-200 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet200", ResNetBottleneck, [3, 24, 36, 3], get_inplanes(), pretrained, progress, **kwargs)
