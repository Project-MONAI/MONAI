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
import torch.nn.functional as F
from torch.fft import fftn
from torch.nn.modules.loss import _Loss

from monai.utils import LossReduction


class JukeboxLoss(_Loss):
    """
    Calculate spectral component based on the magnitude of Fast Fourier Transform (FFT).

    Based on:
        Dhariwal, et al. 'Jukebox: A generative model for music.' https://arxiv.org/abs/2005.00341

    Args:
        spatial_dims: number of spatial dimensions.
        fft_signal_size: signal size in the transformed dimensions. See torch.fft.fftn() for more information.
        fft_norm: {``"forward"``, ``"backward"``, ``"ortho"``} Specifies the normalization mode in the fft. See
            torch.fft.fftn() for more information.

        reduction: {``"none"``, ``"mean"``, ``"sum"``}
            Specifies the reduction to apply to the output. Defaults to ``"mean"``.

            - ``"none"``: no reduction will be applied.
            - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
            - ``"sum"``: the output will be summed.
    """

    def __init__(
        self,
        spatial_dims: int,
        fft_signal_size: tuple[int] | None = None,
        fft_norm: str = "ortho",
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction).value)

        self.spatial_dims = spatial_dims
        self.fft_signal_size = fft_signal_size
        self.fft_dim = tuple(range(1, spatial_dims + 2))
        self.fft_norm = fft_norm

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_amplitude = self._get_fft_amplitude(target)
        target_amplitude = self._get_fft_amplitude(input)

        # Compute distance between amplitude of frequency components
        # See Section 3.3 from https://arxiv.org/abs/2005.00341
        loss = F.mse_loss(target_amplitude, input_amplitude, reduction="none")

        if self.reduction == LossReduction.MEAN.value:
            loss = loss.mean()
        elif self.reduction == LossReduction.SUM.value:
            loss = loss.sum()
        elif self.reduction == LossReduction.NONE.value:
            pass

        return loss

    def _get_fft_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        """
        Calculate the amplitude of the fourier transformations representation of the images

        Args:
            images: Images that are to undergo fftn

        Returns:
            fourier transformation amplitude
        """
        img_fft = fftn(images, s=self.fft_signal_size, dim=self.fft_dim, norm=self.fft_norm)

        amplitude = torch.sqrt(torch.real(img_fft) ** 2 + torch.imag(img_fft) ** 2)

        return amplitude
