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

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option, optional_import

hf_hub_download, _ = optional_import("huggingface_hub", name="hf_hub_download")
EntryNotFoundError, _ = optional_import("huggingface_hub.utils._errors", name="EntryNotFoundError")

MEDICALNET_HUGGINGFACE_REPO_BASENAME = "TencentMedicalNet/MedicalNet-Resnet"
MEDICALNET_HUGGINGFACE_FILES_BASENAME = "resnet_"

__all__ = [
    "ResNet",
    "ResNetBlock",
    "ResNetBottleneck",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
]

logger = logging.getLogger(__name__)


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
        downsample: nn.Module | partial | None = None,
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
        downsample: nn.Module | partial | None = None,
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
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
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
        bias_downsample: whether to use bias term in the downsampling block when `shortcut_type` is 'B', default to `True`.

    """

    def __init__(
        self,
        block: type[ResNetBlock | ResNetBottleneck] | str,
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: tuple[int] | int = 7,
        conv1_t_stride: tuple[int] | int = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avgp_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,
            stride=conv1_stride,
            padding=tuple(k // 2 for k in conv1_kernel_size),
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
        block: type[ResNetBlock | ResNetBottleneck],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: nn.Module | partial | None = None
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
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
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
    block: type[ResNetBlock | ResNetBottleneck],
    layers: list[int],
    block_inplanes: list[int],
    pretrained: bool | str,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model: ResNet = ResNet(block, layers, block_inplanes, **kwargs)
    if pretrained:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(pretrained, str):
            if Path(pretrained).exists():
                logger.info(f"Loading weights from {pretrained}...")
                model_state_dict = torch.load(pretrained, map_location=device)
            else:
                # Throw error
                raise FileNotFoundError("The pretrained checkpoint file is not found")
        else:
            # Also check bias downsample and shortcut.
            if kwargs.get("spatial_dims", 3) == 3:
                if kwargs.get("n_input_channels", 3) == 1 and kwargs.get("feed_forward", True) is False:
                    search_res = re.search(r"resnet(\d+)", arch)
                    if search_res:
                        resnet_depth = int(search_res.group(1))
                    else:
                        raise ValueError("arch argument should be as 'resnet_{resnet_depth}")

                    # Check model bias_downsample and shortcut_type
                    bias_downsample, shortcut_type = get_medicalnet_pretrained_resnet_args(resnet_depth)
                    if shortcut_type == kwargs.get("shortcut_type", "B") and (
                        bool(bias_downsample) == kwargs.get("bias_downsample", False) if bias_downsample != -1 else True
                    ):
                        # Download the MedicalNet pretrained model
                        model_state_dict = get_pretrained_resnet_medicalnet(
                            resnet_depth, device=device, datasets23=True
                        )
                    else:
                        raise NotImplementedError(
                            f"Please set shortcut_type to {shortcut_type} and bias_downsample to"
                            f"{bool(bias_downsample) if bias_downsample!=-1 else 'True or False'}"
                            f"when using pretrained MedicalNet resnet{resnet_depth}"
                        )
                else:
                    raise NotImplementedError(
                        "Please set n_input_channels to 1"
                        "and feed_forward to False in order to use MedicalNet pretrained weights"
                    )
            else:
                raise NotImplementedError("MedicalNet pretrained weights are only avalaible for 3D models")
        model_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        model.load_state_dict(model_state_dict, strict=True)
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


def get_pretrained_resnet_medicalnet(resnet_depth: int, device: str = "cpu", datasets23: bool = True):
    """
    Donwlad resnet pretrained weights from https://huggingface.co/TencentMedicalNet

    Args:
        resnet_depth: depth of the pretrained model. Supported values are 10, 18, 34, 50, 101, 152 and 200
        device: device on which the returned state dict will be loaded. "cpu" or "cuda" for example.
        datasets23: if True, get the weights trained on more datasets (23).
                    Not all depths are available. If not, standard weights are returned.

    Returns:
        Pretrained state dict

    Raises:
        huggingface_hub.utils._errors.EntryNotFoundError: if pretrained weights are not found on huggingface hub
        NotImplementedError: if `resnet_depth` is not supported
    """

    medicalnet_huggingface_repo_basename = "TencentMedicalNet/MedicalNet-Resnet"
    medicalnet_huggingface_files_basename = "resnet_"
    supported_depth = [10, 18, 34, 50, 101, 152, 200]

    logger.info(
        f"Loading MedicalNet pretrained model from https://huggingface.co/{medicalnet_huggingface_repo_basename}{resnet_depth}"
    )

    if resnet_depth in supported_depth:
        filename = (
            f"{medicalnet_huggingface_files_basename}{resnet_depth}.pth"
            if not datasets23
            else f"{medicalnet_huggingface_files_basename}{resnet_depth}_23dataset.pth"
        )
        try:
            pretrained_path = hf_hub_download(
                repo_id=f"{medicalnet_huggingface_repo_basename}{resnet_depth}", filename=filename
            )
        except Exception:
            if datasets23:
                logger.info(f"{filename} not available for resnet{resnet_depth}")
                filename = f"{medicalnet_huggingface_files_basename}{resnet_depth}.pth"
                logger.info(f"Trying with {filename}")
                pretrained_path = hf_hub_download(
                    repo_id=f"{medicalnet_huggingface_repo_basename}{resnet_depth}", filename=filename
                )
            else:
                raise EntryNotFoundError(
                    f"{filename} not found on {medicalnet_huggingface_repo_basename}{resnet_depth}"
                ) from None
        checkpoint = torch.load(pretrained_path, map_location=torch.device(device))
    else:
        raise NotImplementedError("Supported resnet_depth are: [10, 18, 34, 50, 101, 152, 200]")
    logger.info(f"{filename} downloaded")
    return checkpoint.get("state_dict")


def get_medicalnet_pretrained_resnet_args(resnet_depth: int):
    """
    Return correct shortcut_type and bias_downsample
    for pretrained MedicalNet weights according to resnet depth
    """
    # After testing
    # False: 10, 50, 101, 152, 200
    # Any: 18, 34
    bias_downsample = -1 if resnet_depth in [18, 34] else 0  # 18, 10, 34
    shortcut_type = "A" if resnet_depth in [18, 34] else "B"
    return bias_downsample, shortcut_type
