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

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.apps.reconstruction.networks.nets.complex_unet import ComplexUnet
from monai.apps.reconstruction.networks.nets.utils import (
    reshape_batch_channel_to_channel_dim,
    reshape_channel_to_batch_dim,
)
from monai.networks.blocks.fft_utils_t import ifftn_centered_t


class CoilSensitivityModel(nn.Module):
    """
    This class uses a convolutional model to learn coil sensitivity maps for multi-coil MRI reconstruction.
    The convolutional model is :py:class:`monai.apps.reconstruction.networks.nets.complex_unet` by default
    but can be specified by the user as well. Learning is done on the center of the under-sampled
    kspace (that region is fully sampled).

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
        coil_dim: coil dimension in the data
        conv_net: the learning model used to estimate the coil sensitivity maps. default
            is :py:class:`monai.apps.reconstruction.networks.nets.complex_unet`. The only
            requirement on the model is to have 2 as input and output number of channels.
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
        coil_dim: int = 1,
        conv_net: Optional[nn.Module] = None,
    ):
        super().__init__()
        if conv_net is None:
            self.conv_net = ComplexUnet(
                spatial_dims=spatial_dims,
                features=features,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                upsample=upsample,
            )
        else:
            params = [p.shape for p in conv_net.parameters()]
            if (params[0][1], params[-1][0]) != (2, 2):
                raise ValueError(
                    f"(in_channels,out_channels) should be (2,2) but they are ({params[0][1], params[-1][0]})."
                )
            self.conv_net = conv_net  # type: ignore
        self.spatial_dims = spatial_dims
        self.coil_dim = coil_dim

    def get_fully_sampled_region(self, mask: Tensor) -> Sequence[int]:
        """
        Extracts the size of the fully-sampled part of the kspace. Note that when a kspace
        is under-sampled, a part of its center is fully sampled. This part is called the Auto
        Calibration Region (ACR). ACR is used for sensitivity map computation.

        Args:
            mask: the under-sampling mask of shape (..., S, 1) where S denotes the sampling dimension

        Returns:
            A tuple containing
                (1) left index of the region
                (2) right index of the region

        Note:
            Suppose the mask is of shape (1,1,20,1). If this function returns 8,12 as left and right
                indices, then it means that the fully-sampled center region has size 4 starting from 8 to 12.
        """
        left = right = mask.shape[-2] // 2
        while mask[..., right, :]:
            right += 1

        while mask[..., left, :]:
            left -= 1

        return left + 1, right

    def forward(self, masked_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            masked_kspace: the under-sampled kspace (which is the input measurement). Its shape
                is (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data.
            mask: the under-sampling mask with shape (1,1,1,W,1) for 2D data or (1,1,1,1,D,1) for 3D data.

        Returns:
            predicted coil sensitivity maps with shape (B,C,H,W,2) for 2D data or (B,C,H,W,D,2) for 3D data.
        """
        left, right = self.get_fully_sampled_region(mask)
        num_low_freqs = right - left  # size of the fully-sampled center

        # take out the fully-sampled region and set the rest of the data to zero
        x = torch.zeros_like(masked_kspace)
        start = (mask.shape[-2] - num_low_freqs + 1) // 2  # this marks the start of center extraction
        x[..., start : start + num_low_freqs] = masked_kspace[..., start : start + num_low_freqs]

        # apply inverse fourier to the extracted fully-sampled data
        x = ifftn_centered_t(x, spatial_dims=self.spatial_dims)

        x, b = reshape_channel_to_batch_dim(x)  # shape of x will be (B*C,1,...)
        x = self.conv_net(x)
        x = reshape_batch_channel_to_channel_dim(x, b)  # shape will be (B,C,...)
        # normalize the maps
        x /= root_sum_of_squares(x, spatial_dim=self.coil_dim).unsqueeze(self.coil_dim)  # type: ignore
        return x
