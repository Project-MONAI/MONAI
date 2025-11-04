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

from typing import Sequence, Tuple

import torch
from torch import nn

__all__ = ["VectorQuantizer", "EMAQuantizer"]


class EMAQuantizer(nn.Module):
    """
    Vector Quantization module using Exponential Moving Average (EMA) to learn the codebook parameters based on  Neural
    Discrete Representation Learning by Oord et al. (https://arxiv.org/abs/1711.00937) and the official implementation
    that can be found at https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L148 and commit
    58d9a2746493717a7c9252938da7efa6006f3739.

    This module is not compatible with TorchScript while working in a Distributed Data Parallelism Module. This is due
    to lack of TorchScript support for torch.distributed module as per https://github.com/pytorch/pytorch/issues/41353
    on 22/10/2022. If you want to TorchScript your model, please turn set `ddp_sync` to False.

    Args:
        spatial_dims: number of spatial dimensions of the input.
        num_embeddings: number of atomic elements in the codebook.
        embedding_dim: number of channels of the input and atomic elements.
        commitment_cost: scaling factor of the MSE loss between input and its quantized version. Defaults to 0.25.
        decay: EMA decay. Defaults to 0.99.
        epsilon: epsilon value. Defaults to 1e-5.
        embedding_init: initialization method for the codebook. Defaults to "normal".
        ddp_sync: whether to synchronize the codebook across processes. Defaults to True.
    """

    def __init__(
        self,
        spatial_dims: int,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        embedding_init: str = "normal",
        ddp_sync: bool = True,
    ):
        super().__init__()
        self.spatial_dims: int = spatial_dims
        self.embedding_dim: int = embedding_dim
        self.num_embeddings: int = num_embeddings

        assert self.spatial_dims in [2, 3], ValueError(
            f"EMAQuantizer only supports 4D and 5D tensor inputs but received spatial dims {spatial_dims}."
        )

        self.embedding: torch.nn.Embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        if embedding_init == "normal":
            # Initialization is passed since the default one is normal inside the nn.Embedding
            pass
        elif embedding_init == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.embedding.weight.data, mode="fan_in", nonlinearity="linear")
        self.embedding.weight.requires_grad = False

        self.commitment_cost: float = commitment_cost

        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())
        # declare types for mypy
        self.ema_cluster_size: torch.Tensor
        self.ema_w: torch.Tensor
        self.decay: float = decay
        self.epsilon: float = epsilon

        self.ddp_sync: bool = ddp_sync

        # Precalculating required permutation shapes
        self.flatten_permutation = [0] + list(range(2, self.spatial_dims + 2)) + [1]
        self.quantization_permutation: Sequence[int] = [0, self.spatial_dims + 1] + list(
            range(1, self.spatial_dims + 1)
        )

    def quantize(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given an input it projects it to the quantized space and returns additional tensors needed for EMA loss.

        Args:
            inputs: Encoding space tensors of shape [B, C, H, W, D].

        Returns:
            torch.Tensor: Flatten version of the input of shape [B*H*W*D, C].
            torch.Tensor: One-hot representation of the quantization indices of shape [B*H*W*D, self.num_embeddings].
            torch.Tensor: Quantization indices of shape [B,H,W,D,1]

        """
        with torch.autocast("cuda", enabled=False):
            encoding_indices_view = list(inputs.shape)
            del encoding_indices_view[1]

            inputs = inputs.float()

            # Converting to channel last format
            flat_input = inputs.permute(self.flatten_permutation).contiguous().view(-1, self.embedding_dim)

            # Calculate Euclidean distances
            distances = (
                (flat_input**2).sum(dim=1, keepdim=True)
                + (self.embedding.weight.t() ** 2).sum(dim=0, keepdim=True)
                - 2 * torch.mm(flat_input, self.embedding.weight.t())
            )

            # Mapping distances to indexes
            encoding_indices = torch.max(-distances, dim=1)[1]
            encodings = torch.nn.functional.one_hot(encoding_indices, self.num_embeddings).float()

            # Quantize and reshape
            encoding_indices = encoding_indices.view(encoding_indices_view)

            return flat_input, encodings, encoding_indices

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        """
        Given encoding indices of shape [B,D,H,W,1] embeds them in the quantized space
        [B, D, H, W, self.embedding_dim] and reshapes them to [B, self.embedding_dim, D, H, W] to be fed to the
        decoder.

        Args:
            embedding_indices: Tensor in channel last format which holds indices referencing atomic
                elements from self.embedding

        Returns:
            torch.Tensor: Quantize space representation of encoding_indices in channel first format.
        """
        with torch.autocast("cuda", enabled=False):
            embedding: torch.Tensor = (
                self.embedding(embedding_indices).permute(self.quantization_permutation).contiguous()
            )
            return embedding

    def distributed_synchronization(self, encodings_sum: torch.Tensor, dw: torch.Tensor) -> None:
        """
        TorchScript does not support torch.distributed.all_reduce. This function is a bypassing trick based on the
        example: https://pytorch.org/docs/stable/generated/torch.jit.unused.html#torch.jit.unused

        Args:
            encodings_sum: The summation of one hot representation of what encoding was used for each
                position.
            dw: The multiplication of the one hot representation of what encoding was used for each
                position with the flattened input.

        Returns:
            None
        """
        if self.ddp_sync and torch.distributed.is_initialized():
            torch.distributed.all_reduce(tensor=encodings_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(tensor=dw, op=torch.distributed.ReduceOp.SUM)
        else:
            pass

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_input, encodings, encoding_indices = self.quantize(inputs)
        quantized = self.embed(encoding_indices)

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                dw = torch.mm(encodings.t(), flat_input)

                if self.ddp_sync:
                    self.distributed_synchronization(encodings_sum, dw)

                self.ema_cluster_size.data.mul_(self.decay).add_(torch.mul(encodings_sum, 1 - self.decay))

                # Laplace smoothing of the cluster size
                n = self.ema_cluster_size.sum()
                weights = (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
                self.ema_w.data.mul_(self.decay).add_(torch.mul(dw, 1 - self.decay))
                self.embedding.weight.data.copy_(self.ema_w / weights.unsqueeze(1))

        # Encoding Loss
        loss = self.commitment_cost * torch.nn.functional.mse_loss(quantized.detach(), inputs)

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()

        return quantized, loss, encoding_indices


class VectorQuantizer(torch.nn.Module):
    """
    Vector Quantization wrapper that is needed as a workaround for the AMP to isolate the non fp16 compatible parts of
    the quantization in their own class.

    Args:
        quantizer (torch.nn.Module):  Quantizer module that needs to return its quantized representation, loss and index
            based quantized representation.
    """

    def __init__(self, quantizer: EMAQuantizer):
        super().__init__()

        self.quantizer: EMAQuantizer = quantizer

        self.perplexity: torch.Tensor = torch.rand(1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized, loss, encoding_indices = self.quantizer(inputs)
        # Perplexity calculations
        avg_probs = (
            torch.histc(encoding_indices.float(), bins=self.quantizer.num_embeddings, max=self.quantizer.num_embeddings)
            .float()
            .div(encoding_indices.numel())
        )

        self.perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized

    def embed(self, embedding_indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.embed(embedding_indices=embedding_indices)

    def quantize(self, encodings: torch.Tensor) -> torch.Tensor:
        output = self.quantizer(encodings)
        encoding_indices: torch.Tensor = output[2]
        return encoding_indices
