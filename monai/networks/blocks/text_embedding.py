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


class TextEncoder(nn.Module):
    """
    Text to vision encoding by Contrastive Language-Image Pre-training (CLIP) or random embedding.
    The text to vision encoder loads the pre-trained or random initialized weights with connection to 2D/3D vision models.

    Contrastive Language-Image Pre-training (CLIP), based on: "Radford et al.,
    Learning Transferable Visual Models From Natural Language Supervision <https://arxiv.org/abs/2103.00020>"

    Connecting text and medical 3D image, based on: "Liu et al.,
    CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection <https://arxiv.org/pdf/2301.00785.pdf>"
    """
    def __init__(
        self,
        out_channels: int,
        text_dim: int = 512,
        hidden_size: int = 256,
        encoding: str = "clip_embedding",
    ) -> None:
        """
        Args:
            out_channels: number of output channels, to control text-baesd embedding for classes.
            text_dim: dimension of text embeddings.
            hidden_size: dimension of hidden features, compatible to different vision feature dimensions.
            encoding: the text embedding type, default to use clip text pretrained weights
        """
        super().__init__()
        self.encoding = encoding

        if self.encoding == 'rand_embedding':
            self.text_embedding = nn.Embedding(out_channels, hidden_size)
        elif self.encoding == 'clip_embedding':
            self.register_buffer('text_embedding', torch.randn(out_channels, text_dim))
            self.text_to_vision = nn.Linear(text_dim, hidden_size)
        else:
            raise Exception(f'{self.encoding} is not implemented, please add your own')

    def forward(self):
        if self.encoding == 'clip_embedding':
            test_encoding = nn.functional.relu(self.text_to_vision(self.text_embedding))
            test_encoding = test_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        else:
            # text embedding as random initialized 'rand_embedding'
            test_encoding = self.text_embedding.weight.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        return test_encoding
