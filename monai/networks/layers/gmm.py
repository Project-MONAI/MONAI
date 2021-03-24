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

from monai._extensions.loader import load_module

__all__ = ["GaussianMixtureModel"]


class GaussianMixtureModel(torch.nn.Module):
    """
    Takes an initial labeling and uses a mixture of gaussians to approximate each classes
    distribution in the feature space. Each unlabled element is then asigned a probability
    of belonging to each class based on it's fit to each classes approximate distribution.

    See:
        https://en.wikipedia.org/wiki/Mixture_model

    Args:
        channel_count (int): The number of features per element.
        mixture_count (int): The number of class distibutions.
        mixture_size (int): The number gaussian components per class distribution.
        features (torch.Tensor): features for each element.
        initial_labels (torch.Tensor): initial labeling for each element.

    Returns:
        output_logits (torch.Tensor): class assignment probabilities for each element.
    """

    def __init__(self, channel_count, mixture_count, mixture_size):
        super(GaussianMixtureModel, self).__init__()
        self.compiled_extention = load_module(
            "gmm", {"CHANNEL_COUNT": channel_count, "MIXTURE_COUNT": mixture_count, "MIXTURE_SIZE": mixture_size}
        )
        self.channel_count = channel_count
        self.mixture_count = mixture_count
        self.mixture_size = mixture_size

    def forward(self, features, initial_labels):

        assert features.size(1) == self.channel_count

        output_logits = self.compiled_extention.gmm(features, initial_labels)

        return output_logits
