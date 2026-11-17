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

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

from monai.apps.utils import download_url
from monai.utils import optional_import

torchvision, _ = optional_import("torchvision")

__all__ = ["LPIPS", "LPIPSFeatures", "NetLinLayer", "ScalingLayer"]

# Resolve the default at construction time so torch.hub.set_dir() remains effective.
LPIPS_CHECKPOINTS_DIR: Path | None = None
LPIPS_WEIGHTS_URL = "https://huggingface.co/MONAI/checkpoints/resolve/main"
LPIPS_NETWORKS = {
    "alex": (
        "alexnet",
        "AlexNet_Weights",
        (2, 5, 8, 10, 12),
        (64, 192, 384, 256, 256),
        "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
    ),
    "vgg": (
        "vgg16",
        "VGG16_Weights",
        (4, 9, 16, 23, 30),
        (64, 128, 256, 512, 512),
        "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868",
    ),
    "squeeze": (
        "squeezenet1_1",
        "SqueezeNet1_1_Weights",
        (2, 5, 8, 10, 11, 12, 13),
        (64, 128, 256, 384, 384, 512, 512),
        "4a5350f23600cb79923ce65bb07cbf57dca461329894153e05a1346bd531cf76",
    ),
}


class ScalingLayer(nn.Module):
    """Apply the fixed RGB shift and scale used by LPIPS v0.1."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Scale an image batch, broadcasting grayscale inputs to RGB.

        Args:
            input: Image tensor of shape ``(B, C, H, W)``, with one or three channels.

        Returns:
            Scaled image tensor of shape ``(B, 3, H, W)``.
        """
        return (input - self.shift) / self.scale


class LPIPSFeatures(nn.Sequential):
    """Extract frozen Torchvision features using legacy LPIPS slice names.

    Args:
        features: Ordered feature layers from a Torchvision backbone.
        endpoints: Exclusive layer indices at which each successive slice ends.
    """

    def __init__(self, features: nn.Sequential, endpoints: tuple[int, ...]) -> None:
        slices: OrderedDict[str, nn.Module] = OrderedDict()
        start = 0
        for index, end in enumerate(endpoints, 1):
            # Keep the original names so existing PerceptualLoss checkpoints can be loaded.
            block = nn.Sequential()
            for layer in range(start, end):
                block.add_module(str(layer), features[layer])
            slices[f"slice{index}"] = block
            start = end
        super().__init__(slices)
        self.requires_grad_(False)

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Collect the output of each feature slice.

        Args:
            input: Scaled RGB image tensor of shape ``(B, 3, H, W)``.

        Returns:
            Feature tensors in slice order, each with shape ``(B, C_i, H_i, W_i)``.
        """
        outputs = []
        for layer in self:
            # Each slice consumes the preceding features, not the original RGB input.
            input = layer(input)
            outputs.append(input)
        return tuple(outputs)


class NetLinLayer(nn.Sequential):
    """Calibrate feature differences with Dropout and a bias-free 1x1 convolution.

    Args:
        channels: Number of input feature channels.
    """

    def __init__(self, channels: int) -> None:
        # The named child retains the model.1.weight keys in upstream calibration files
        # and saved PerceptualLoss checkpoints while Sequential supplies forward().
        super().__init__(OrderedDict(model=nn.Sequential(nn.Dropout(), nn.Conv2d(channels, 1, 1, bias=False))))


class LPIPS(nn.Module):
    """LPIPS v0.1 network used by :class:`monai.losses.PerceptualLoss`.

    ImageNet feature weights come from Torchvision. ``pretrained`` controls only the
    learned linear calibration, matching the original LPIPS constructor. The v0.1
    calibration weights are downloaded from the official ``MONAI/checkpoints`` repository
    on Hugging Face, verified with SHA-256, and cached in the Torch Hub checkpoints directory.

    Args:
        pretrained: Whether to load pretrained linear calibration weights. The feature
            backbone always uses ImageNet weights, including when this is ``False``.
        net: Feature backbone name: ``"alex"``, ``"vgg"``, or ``"squeeze"``.

    Raises:
        ValueError: If ``net`` is not a supported feature backbone name.
        monai.apps.HashCheckError: If the calibration checkpoint fails integrity verification.
        RuntimeError: If downloading the calibration checkpoint fails.
        OSError: If the calibration checkpoint cannot be downloaded or read.
    """

    def __init__(self, pretrained: bool = True, net: str = "alex") -> None:
        super().__init__()
        if net not in LPIPS_NETWORKS:
            raise ValueError(f"Unsupported LPIPS network: {net}.")
        factory, weights, endpoints, channels, hash_val = LPIPS_NETWORKS[net]
        self.scaling_layer = ScalingLayer()
        backbone = getattr(torchvision.models, factory)(weights=getattr(torchvision.models, weights).IMAGENET1K_V1)
        self.net = LPIPSFeatures(backbone.features, endpoints)
        # Legacy checkpoints contain both lin{index} and lins.{index} keys, so register
        # each calibration layer under both names while sharing the same parameters.
        layers = []
        for index, channel_count in enumerate(channels):
            layer = NetLinLayer(channel_count)
            self.add_module(f"lin{index}", layer)
            layers.append(layer)
        self.lins = nn.ModuleList(layers)
        if pretrained:
            directory = (
                LPIPS_CHECKPOINTS_DIR if LPIPS_CHECKPOINTS_DIR is not None else Path(torch.hub.get_dir(), "checkpoints")
            )
            path = directory / f"lpips_v0.1_{net}.pth"
            download_url(
                f"{LPIPS_WEIGHTS_URL}/{net}.pth", filepath=path, hash_val=hash_val, hash_type="sha256", progress=False
            )
            self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=False)
        self.eval()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute one LPIPS distance for each image pair.

        Args:
            input: Image tensor of shape ``(B, C, H, W)``, with one or three channels.
            target: Reference image tensor with the same shape as ``input``.

        Returns:
            Per-image distances of shape ``(B, 1, 1, 1)``, before the outer
            ``PerceptualLoss`` reduction.
        """
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
