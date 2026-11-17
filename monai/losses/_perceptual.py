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

# Adapted from richzhang/PerceptualSimilarity, commit
# 082bb24f84c091ea94de2867d34c4544f68e0963 (lpips/lpips.py and pretrained_networks.py).
# Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros,
# Eli Shechtman, Oliver Wang. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from monai.utils import optional_import

torchvision, _ = optional_import("torchvision")


class _ScalingLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input - self.shift) / self.scale


class _LPIPSFeatures(nn.Module):
    def __init__(self, features: nn.Sequential, endpoints: tuple[int, ...]) -> None:
        super().__init__()
        self.num_slices = len(endpoints)
        start = 0
        for index, end in enumerate(endpoints, 1):
            # Keep the original names so existing PerceptualLoss checkpoints can be loaded.
            block = nn.Sequential()
            for layer in range(start, end):
                block.add_module(str(layer), features[layer])
            self.add_module(f"slice{index}", block)
            start = end
        self.requires_grad_(False)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = []
        for index in range(1, self.num_slices + 1):
            input = getattr(self, f"slice{index}")(input)
            outputs.append(input)
        return tuple(outputs)


class _NetLinLayer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Dropout(), nn.Conv2d(channels, 1, 1, bias=False))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)


class LPIPS(nn.Module):
    """Internal LPIPS v0.1 implementation for :class:`monai.losses.PerceptualLoss`.

    ImageNet feature weights come from Torchvision. ``pretrained`` controls only the
    learned linear calibration, matching the original LPIPS constructor. The v0.1
    calibration weights and their license are distributed in the adjacent weights directory.
    """

    def __init__(self, pretrained: bool = True, net: str = "alex") -> None:
        super().__init__()
        networks = {
            "alex": ("alexnet", "AlexNet_Weights", (2, 5, 8, 10, 12), (64, 192, 384, 256, 256)),
            "vgg": ("vgg16", "VGG16_Weights", (4, 9, 16, 23, 30), (64, 128, 256, 512, 512)),
            "squeeze": (
                "squeezenet1_1",
                "SqueezeNet1_1_Weights",
                (2, 5, 8, 10, 11, 12, 13),
                (64, 128, 256, 384, 384, 512, 512),
            ),
        }
        if net not in networks:
            raise ValueError(f"Unsupported LPIPS network: {net}.")
        factory, weights, endpoints, channels = networks[net]
        self.scaling_layer = _ScalingLayer()
        backbone = getattr(torchvision.models, factory)(weights=getattr(torchvision.models, weights).IMAGENET1K_V1)
        self.net = _LPIPSFeatures(backbone.features, endpoints)
        layers = []
        for index, channel_count in enumerate(channels):
            layer = _NetLinLayer(channel_count)
            self.add_module(f"lin{index}", layer)
            layers.append(layer)
        self.lins = nn.ModuleList(layers)
        if pretrained:
            path = Path(__file__).with_name("weights") / f"{net}.pth"
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
        self.eval()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        features_input = self.net(self.scaling_layer(input))
        features_target = self.net(self.scaling_layer(target))
        results = []
        for layer, source, reference in zip(self.lins, features_input, features_target):
            # LPIPS places epsilon outside the square root. Keep its original normalization
            # rather than the different normalization used by the other perceptual networks.
            source = source / (torch.sqrt(torch.sum(source**2, dim=1, keepdim=True)) + 1e-10)
            reference = reference / (torch.sqrt(torch.sum(reference**2, dim=1, keepdim=True)) + 1e-10)
            results.append(layer((source - reference) ** 2).mean([2, 3], keepdim=True))
        result = results[0]
        for value in results[1:]:
            result = result + value
        return result
