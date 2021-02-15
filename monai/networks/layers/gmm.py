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

import torch

from monai.utils.module import optional_import

_C, _ = optional_import("monai._C")

__all__ = ["GaussianMixtureModel"]

class GaussianMixtureModel(torch.nn.Module):
    """
    Fits data using a mixture of gaussians using expectation maximization.

    See:
        https://en.wikipedia.org/wiki/Mixture_model

    Args:
        input_tensor (torch.Tensor): input tensor.
        label_tensor (torch.Tensor): initial pixel-wise labeling

    Returns:
        output (torch.Tensor): output tensor.
    """

    def __init__(self, mixture_count, gaussians_per_mixture):
        super(GaussianMixtureModel, self).__init__()
        self.mixture_count = mixture_count 
        self.gaussians_per_mixture = gaussians_per_mixture

    def forward(self, input_tensor, label_tensor):
        output = _C.gmm(input_tensor, label_tensor, self.mixture_count, self.gaussians_per_mixture)
        return output
