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
from torch.utils import model_zoo

url_map = {
    "clip_encoding_universal_model_32": (
        "https://github.com/Project-MONAI/MONAI-extra-test-data/"
        "releases/download/0.8.1/clip_encoding_universal_model.pth"
    )
}


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
        spatial_dims: int = 3,
        text_dim: int = 512,
        hidden_size: int = 256,
        encoding: str = "clip_encoding_universal_model_32",
        pretrained: bool = True,
    ) -> None:
        """
        Args:
            out_channels: number of output channels, to control text-based embedding for classes.
            spatial_dims: number of spatial dims.
            text_dim: dimension of text embeddings.
            hidden_size: dimension of hidden features, compatible to different vision feature dimensions.
            encoding: the text embedding type, default to use clip text pretrained weights.
            pretrained: whether to load pretrained weights from e.g., (CLIP) to initialize text embeddings, default to False.
        """
        super().__init__()
        self.encoding = encoding

        self.spatial_dims = spatial_dims
        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if self.encoding == "rand_embedding":
            self.text_embedding = nn.Embedding(out_channels, hidden_size)
        else:
            self.register_buffer("text_embedding", torch.randn(out_channels, text_dim))

            if pretrained:
                model_url = url_map[self.encoding]
                pretrain_state_dict = model_zoo.load_url(model_url, map_location="cpu")
                self.text_embedding.data = pretrain_state_dict.float()  # type: ignore
            else:
                print(f"{self.encoding} is not implemented, and can not be downloaded, please load your own")

            self.text_to_vision = nn.Linear(text_dim, hidden_size)

    def forward(self):
        if self.encoding == "rand_embedding":
            # text embedding as random initialized 'rand_embedding'
            text_embedding = self.text_embedding.weight
        else:
            print(self.text_embedding)
            text_embedding = nn.functional.relu(self.text_to_vision(self.text_embedding))

        if self.spatial_dims == 3:
            text_embedding = text_embedding.unsqueeze(2).unsqueeze(2).unsqueeze(2)
        elif self.spatial_dims == 2:
            text_embedding = text_embedding.unsqueeze(2).unsqueeze(2)

        return text_embedding
