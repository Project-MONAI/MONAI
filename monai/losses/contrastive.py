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

from typing import Union

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction


class ContrastiveLoss(_Loss):

    """
    Compute the Contrastive loss defined in:

        Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International
        conference on machine learning. PMLR, 2020. (http://proceedings.mlr.press/v119/chen20j.html)

    Adapted from:
        https://github.com/Sara-Ahmed/SiT/blob/1aacd6adcd39b71efc903d16b4e9095b97dda76f/losses.py#L5

    """

    def __init__(
        self, temperature: float = 0.5, batch_size: int = 1, reduction: Union[LossReduction, str] = LossReduction.SUM
    ) -> None:
        """
        Args:
            temperature: Can be scaled between 0 and 1 for learning from negative samples, ideally set to 0.5.

        Raises:
            AssertionError: When an input of dimension length > 2 is passed
            AssertionError: When input and target are of different shapes

        """
        super().__init__(reduction=LossReduction(reduction).value)

        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be B[F].
            target: the shape should be B[F].

        Raises:
            ValueError: When ``self.reduction`` is not one of ["sum", "none"].
        """
        if len(target.shape) > 2 or len(input.shape) > 2:
            raise AssertionError(
                f"Either target or input has dimensions greater than 2 where target "
                f"shape is ({target.shape}) and input shape is ({input.shape})"
            )

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        temperature_tensor = torch.tensor(self.temperature).to(input.device)

        norm_i = F.normalize(input, dim=1)
        norm_j = F.normalize(target, dim=1)

        negatives_mask = ~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=torch.bool)
        negatives_mask = torch.tensor(negatives_mask, dtype=torch.float)
        negatives_mask = torch.clone(torch.as_tensor(negatives_mask)).to(input.device)

        repr = torch.cat([norm_i, norm_j], dim=0)
        sim_matrix = F.cosine_similarity(repr.unsqueeze(1), repr.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim_matrix, self.batch_size)
        sim_ji = torch.diag(sim_matrix, -self.batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / temperature_tensor)
        denominator = negatives_mask * torch.exp(sim_matrix / temperature_tensor)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(loss_partial) / (2 * self.batch_size)
        raise ValueError(f"Unsupported reduction: {self.reduction}, " f'available options are ["mean", "sum", "none"].')
