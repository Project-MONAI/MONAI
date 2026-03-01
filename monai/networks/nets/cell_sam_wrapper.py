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

import torch
from torch import nn
from torch.nn import functional as F

from monai.utils import optional_import

build_sam_vit_b, has_sam = optional_import("segment_anything.build_sam", name="build_sam_vit_b")

_all__ = ["CellSamWrapper"]


class CellSamWrapper(torch.nn.Module):
    """
    CellSamWrapper is thin wrapper around SAM model https://github.com/facebookresearch/segment-anything
    with an image only decoder, that can be used for segmentation tasks.


    Args:
        auto_resize_inputs: whether to resize inputs before passing to the network.
            (usually they need be resized, unless they are already at the expected size)
        network_resize_roi: expected input size for the network.
            (currently SAM expects 1024x1024)
        checkpoint: checkpoint file to load the SAM weights from.
            (this can be downloaded from SAM repo https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
        return_features: whether to return features from SAM encoder
            (without using decoder/upsampling to the original input size)

    """

    def __init__(
        self,
        auto_resize_inputs=True,
        network_resize_roi=(1024, 1024),
        checkpoint="sam_vit_b_01ec64.pth",
        return_features=False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.network_resize_roi = network_resize_roi
        self.auto_resize_inputs = auto_resize_inputs
        self.return_features = return_features

        if not has_sam:
            raise ValueError(
                "SAM is not installed, please run: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        model = build_sam_vit_b(checkpoint=checkpoint)

        model.prompt_encoder = None
        model.mask_decoder = None

        model.mask_decoder = nn.Sequential(
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
        )

        self.model = model

    def forward(self, x):
        sh = x.shape[2:]

        if self.auto_resize_inputs:
            x = F.interpolate(x, size=self.network_resize_roi, mode="bilinear")

        x = self.model.image_encoder(x)

        if not self.return_features:
            x = self.model.mask_decoder(x)
            if self.auto_resize_inputs:
                x = F.interpolate(x, size=sh, mode="bilinear")

        return x
