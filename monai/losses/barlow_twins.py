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
from torch.nn.modules.loss import _Loss


class BarlowTwinsLoss(_Loss):
    """
    The Barlow Twins cost function takes the representations extracted by a neural network from two
    distorted views and seeks to make the cross-correlation matrix of the two representations tend
    towards identity. This encourages the neural network to learn similar representations with the least
    amount of redundancy. This cost function can be used in particular in multimodal learning to work on
    representations from two modalities. The most common use case is for unsupervised learning, where data
    augmentations are used to generate 2 distorted views of the same sample to force the encoder to
    extract useful features for downstream tasks.

    Zbontar, Jure, et al. "Barlow Twins: Self-Supervised Learning via Redundancy Reduction" International
    conference on machine learning. PMLR, 2020. (http://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf)

    Adapted from:
        https://github.com/facebookresearch/barlowtwins

    """

    def __init__(self, lambd: float = 5e-3) -> None:
        """
        Args:
            lamb: Can be any float to handle the informativeness and invariance trade-off. Ideally set to 5e-3.

        Raises:
            ValueError: When an input of dimension length > 2 is passed
            ValueError: When input and target are of different shapes
            ValueError: When batch size is less than or equal to 1

        """
        super().__init__()
        self.lambd = lambd

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be B[F].
            target: the shape should be B[F].
        """
        if len(target.shape) > 2 or len(input.shape) > 2:
            raise ValueError(
                f"Either target or input has dimensions greater than 2 where target "
                f"shape is ({target.shape}) and input shape is ({input.shape})"
            )

        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        if target.size(0) <= 1:
            raise ValueError(
                f"Batch size must be greater than 1 to compute Barlow Twins Loss, but got {target.size(0)}"
            )

        lambd_tensor = torch.as_tensor(self.lambd).to(input.device)
        batch_size = input.shape[0]

        # normalize input and target
        input_norm = (input - input.mean(0)) / input.std(0).add(1e-6)
        target_norm = (target - target.mean(0)) / target.std(0).add(1e-6)

        # cross-correlation matrix
        c = torch.mm(input_norm.t(), target_norm) / batch_size  # input_norm.t() is FxB, target_norm is BxF so c is FxF

        # loss
        c_diff = (c - torch.eye(c.size(0), device=c.device)).pow_(2)  # FxF
        c_diff[~torch.eye(c.size(0), device=c.device).bool()] *= lambd_tensor

        return c_diff.sum()
