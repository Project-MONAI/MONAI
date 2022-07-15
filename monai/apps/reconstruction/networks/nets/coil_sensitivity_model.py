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

from typing import Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.apps.reconstruction.networks.nets.complex_unet import ComplexUnet
from monai.networks.blocks.fft_utils_t import ifftn_centered_t


class CoilSensitivityModel(nn.Module):
    """
    This class uses :py:class:`monai.apps.reconstruction.networks.nets.complex_unet` to learn
    coil sensitivity maps for multi-coil MRI reconstruction. Learning is done on the center of
    the under-sampled kspace (that region is fully sampled).

    The data being a (complex) 2-channel tensor is a requirement for using this model.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        spatial_dims: number of spatial dimensions.
        features: six integers as numbers of features. denotes number of channels in each layer.
        act: activation type and arguments. Defaults to LeakyReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        bias: whether to have a bias term in convolution blocks. Defaults to True.
        dropout: dropout ratio. Defaults to 0.0.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
    ):
        super().__init__()
        self.unet = ComplexUnet(
            spatial_dims=spatial_dims,
            features=features,
            act=act,
            norm=norm,
            bias=bias,
            dropout=dropout,
            upsample=upsample,
        )

    def chans_to_batch_dim(self, x: Tensor) -> Sequence:
        """
        Combines batch and channel dimensions. For example, x with shape(B,C,...)
        will be of shape (B*C,1,...)
        """
        b, c, *other = x.shape
        return x.contiguous().view(b * c, 1, *other), b

    def batch_chans_to_chan_dim(self, x: Tensor, batch_size: int) -> Tensor:
        """
        Detaches batch and channel dimensions. For example, x with shape(B*C,1,...)
        will be of shape (B,C,...)
        """
        bc, one, *other = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, *other)

    def divide_root_sum_of_squares(self, x: Tensor) -> Tensor:
        """
        Divides input by its root sum of squares along the coil dimension (dim=1)
        """
        return x / root_sum_of_squares(x, spatial_dim=1).unsqueeze(1)  # type: ignore

    def forward(self, masked_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            masked_kspace: the under-sampled kspace with shape (B,C,H,W,2)
            mask: the under-sampling mask (1,1,1,W,1)

        Returns:
            predicted coil sensitivity maps (B,C,H,W,2)
        """

        def get_low_frequency_lines(mask: Tensor) -> Sequence[int]:
            """
            Extracts the size of the fully-sampled part of the kspace. Note that when a kspace
            is under-sampled, a part of its center is fully sampled. That part is used for
            sensitivity map computation.

            Args:
                mask: the under-sampling mask
            """
            left = right = mask.shape[-2] // 2
            while mask[..., right, :]:
                right += 1

            while mask[..., left, :]:
                left -= 1

            return left + 1, right

        left, right = get_low_frequency_lines(mask)
        num_low_freqs = right - left  # size of the fully-sampled center
        pad = (mask.shape[-2] - num_low_freqs + 1) // 2

        x = torch.zeros_like(masked_kspace)
        x[:, :, :, pad : pad + num_low_freqs] = masked_kspace[:, :, :, pad : pad + num_low_freqs]

        x = ifftn_centered_t(x, spatial_dims=2)

        x, b = self.chans_to_batch_dim(x)
        x = self.unet(x)
        x = self.batch_chans_to_chan_dim(x, b)
        x = self.divide_root_sum_of_squares(x)
        return x
