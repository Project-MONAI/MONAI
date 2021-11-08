# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Callable, List, Optional, Union

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction

class ContrasiveLoss(_Loss):

    """
    Compute the Contrastive loss defined in:

        Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International
        conference on machine learning. PMLR, 2020. (http://proceedings.mlr.press/v119/chen20j.html)

    Adapted from:
        https://github.com/Sara-Ahmed/SiT/blob/1aacd6adcd39b71efc903d16b4e9095b97dda76f/losses.py#L5

    """

    def __init__(
        self,
        normalize: bool = True,
        temperature: float = 0.5,
        batch_size: int = 1,
    ) -> None:
        """
        Args:
            normalize: If True, input feature vector is normalized along the vector (B, F). F will be normalized
            temperature: Can be scaled between 0 and 1 for learning from negative samples, ideally set to 0.5.

        Raises:
            TypeError: When ``other_act`` is not an ``Optional[Callable]``.
            ValueError: When more than 1 of [``sigmoid=True``, ``softmax=True``, ``other_act is not None``].
                Incompatible values.

        """
        self.batch_size = batch_size
        self.normalize = normalize
        self.temperature = temperature
        self.negatives_mask = torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be B[F].
            target: the shape should be B[F].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["sum", "none"].
        """
        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        if self.normalize:
            norm_i = F.normalize(input, dim=1)
            norm_j = F.normalize(target, dim=1)

        else:
            norm_i = input
            norm_j = target

        repr = torch.cat([norm_i, norm_j], dim=0)
        sim_matrix = F.cosine_similarity(repr.unsqueeze(1), repr.unsqueeze(0), dim=2)

        sim_ij = torch.diag(sim_matrix, self.batch_size)
        sim_ji = torch.diag(sim_matrix, -self.batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss_partial) / (2 * self.batch_size)
        raise ValueError(f'Unsupported reduction: {self.reduction}, '
                         f'available options are ["mean", "sum", "none"].')




