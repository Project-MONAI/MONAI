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

import torch.nn as nn
from torch import Tensor

from monai.apps.reconstruction.complex_utils import complex_abs
from monai.apps.reconstruction.mri_utils import root_sum_of_squares
from monai.apps.reconstruction.networks.blocks.varnetblock import VarNetBlock
from monai.apps.reconstruction.networks.nets.complex_unet import ComplexUnet
from monai.apps.reconstruction.networks.nets.coil_sensitivity_model import CoilSensitivityModel
from monai.networks.blocks.fft_utils_t import ifftn_centered_t


class VariationalNetworkModel(nn.Module):
    """
    The end-to-end variational network (or simply e2e-VarNet) based on Sriram et. al., "End-to-end variational
    networks for accelerated MRI reconstruction".
    It comprises several cascades

    Since the e2e-VarNet is memory-consuming for training, we have added a model parallelism feature which is
    automatically performed when there are more than 1 devices available.

    Modified and adopted from: https://github.com/facebookresearch/fastMRI

    Args:
        hparams: an object containing model and training hyper-parameters such as number of channels
            or list of available devices
    """

    def __init__(self, hparams): #### pass different parts of hparams, separately
        super().__init__()
        self.sens_net = CoilSensitivityModel(features=hparams.sens_features).to(hparams.devices[0])

        self.cascades = nn.ModuleList([VarNetBlock(ComplexUnet(features=hparams.features)) for i in range(hparams.num_cascades)])

        # model parallelism based on the number of devices available
        self.num_cascades_per_device = hparams.num_cascades // len(hparams.devices) + 1
        self.devices = hparams.devices
        for i, cascade in enumerate(self.cascades):
            self.cascades[i] = cascade.to(self.devices[i // self.num_cascades_per_device])

    def forward(self, masked_kspace: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            Masked_kspace: the under-sampled kspace
            mask: the under-sampling mask

        Returns:
            reconstructed image which is the root sum of squares (rss) of the absolute value
            of the inverse fourier of the predicted kspace (note that rss combines coil images into one image).
        """
        sens_maps = self.sens_net(masked_kspace, mask)
        kspace_pred = masked_kspace.clone()

        for i, cascade in enumerate(self.cascades):
            kspace_pred = kspace_pred.to(self.devices[i // self.num_cascades_per_device])
            kspace_pred = cascade(
                kspace_pred,
                masked_kspace.to(self.devices[i // self.num_cascades_per_device]),
                mask.to(self.devices[i // self.num_cascades_per_device]),
                sens_maps.to(self.devices[i // self.num_cascades_per_device]),
            )

            output_image = root_sum_of_squares(
                complex_abs(ifftn_centered_t(kspace_pred, spatial_dims=2)), spatial_dim=1
            )
        return output_image  # type: ignore
