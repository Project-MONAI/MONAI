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

from warnings import warn

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class ContrastiveLoss(_Loss):
    """
    Compute the Contrastive loss defined in:

        Chen, Ting, et al. "A simple framework for contrastive learning of visual representations." International
        conference on machine learning. PMLR, 2020. (http://proceedings.mlr.press/v119/chen20j.html)

    Adapted from:
        https://github.com/Sara-Ahmed/SiT/blob/1aacd6adcd39b71efc903d16b4e9095b97dda76f/losses.py#L5

    The loss is evaluated using positive-shifted logits in log space. By default, float16
    and bfloat16 inputs use float32 arithmetic to preserve precision and accumulation range.

    Very small temperatures can amplify roundoff in cosine-similarity gradients.
    For higher precision, pass both embeddings as float64 before evaluating the loss,
    for example ``loss(input.double(), target.double())``, or set ``compute_dtype=torch.float64``.
    Casting the computed loss to float64 does not improve the precision of its backward computation.

    """

    def __init__(
        self, temperature: float = 0.5, batch_size: int = -1, *, compute_dtype: torch.dtype | None = None
    ) -> None:
        """
        Args:
            temperature: Can be scaled between 0 and 1 for learning from negative samples, ideally set to 0.5.
            batch_size: Legacy parameter. The batch size is estimated dynamically in ``forward``.
            compute_dtype: Floating-point dtype for similarity and loss computations, either ``torch.float32``
                or ``torch.float64``. Defaults to None, preserving float64 inputs and promoting float16/bfloat16
                inputs to float32. Selecting float64 can reduce backward roundoff at very small temperatures
                on devices supporting float64, with additional computation and memory costs. Gradients are
                still stored in the original input dtype; existing input quantization cannot be recovered.

        Raises:
            ValueError: If ``compute_dtype`` is neither None nor a supported computation dtype.

        """
        super().__init__()
        if compute_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("compute_dtype must be None, torch.float32, or torch.float64.")
        self.temperature = temperature
        self.compute_dtype = compute_dtype

        if batch_size != -1:
            warn("batch_size is no longer required to be set. It will be estimated dynamically in the forward call")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be B[F].
            target: the shape should be B[F].

        Returns:
            A scalar loss tensor in ``compute_dtype`` when specified. Otherwise, float16/bfloat16
            inputs use float32, and float32/float64 inputs retain their dtype.

        Raises:
            ValueError: If either input has more than two dimensions or their shapes differ.
        """
        if len(target.shape) > 2 or len(input.shape) > 2:
            raise ValueError(
                f"Either target or input has dimensions greater than 2 where target "
                f"shape is ({target.shape}) and input shape is ({input.shape})"
            )

        if target.shape != input.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from input ({input.shape})")

        batch_size = input.shape[0]

        self_mask = torch.eye(batch_size * 2, dtype=torch.bool, device=input.device)

        repr = torch.cat([input, target], dim=0)
        compute_dtype = self.compute_dtype
        if compute_dtype is not None:
            repr = repr.to(dtype=compute_dtype)
        elif repr.dtype in (torch.float16, torch.bfloat16):
            repr = repr.float()
        temperature_tensor = torch.as_tensor(self.temperature, dtype=repr.dtype, device=input.device)
        sim_matrix = F.cosine_similarity(repr.unsqueeze(1), repr.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim_matrix, batch_size)
        sim_ji = torch.diag(sim_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        # Shift before scaling so the positive logit is zero, avoiding cancellation after logsumexp.
        logits = ((sim_matrix - positives.unsqueeze(1)) / temperature_tensor).masked_fill(self_mask, -torch.inf)

        loss_partial = torch.logsumexp(logits, dim=1)

        return torch.sum(loss_partial) / (2 * batch_size)
