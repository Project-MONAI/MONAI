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

class GaussianMixtureModel():
    """
    Takes an initial labeling and uses a mixture of gaussians to approximate each classes
    distribution in the feature space. Each unlabled element is then asigned a probability
    of belonging to each class based on it's fit to each classes approximated distribution.

    See:
        https://en.wikipedia.org/wiki/Mixture_model
    """

    def __init__(self, channel_count, mixture_count, mixture_size):
        """
        Args:
            channel_count (int): The number of features per element.
            mixture_count (int): The number of class distibutions.
            mixture_size (int): The number gaussian components per class distribution.
        """
        self.channel_count = channel_count
        self.mixture_count = mixture_count
        self.mixture_size = mixture_size
        self.compiled_extention = load_module(
            "gmm", {"CHANNEL_COUNT": channel_count, "MIXTURE_COUNT": mixture_count, "MIXTURE_SIZE": mixture_size}
        )
        self.params, self.scratch = self.compiled_extention.init()

    def reset(self):
        """
        Resets the parameters of the model.
        """
        self.params, self.scratch = self.compiled_extention.init()

    def learn(self, features, labels):
        """
        Learns the distribution of each class from provided labels.

        Args:
            features (torch.Tensor): features for each element.
            initial_labels (torch.Tensor): initial labeling for each element.

        Returns:
            output_logits (torch.Tensor): class assignment probabilities for each element.
        """
        self.compiled_extention.learn(self.params, self.scratch, features, labels)

    def apply(self, features):
        """
        Applys the current model to a set of feature vectors.

        Args:
            features (torch.Tensor): feature vectors for each element.

        Returns:
            output (torch.Tensor): class assignment probabilities for each element.
        """
        return self.compiled_extention.apply(self.params, features)


class _LearnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, compiled_extention, params, scratch, features, labels, learn_rate):
        return compiled_extention.learn(params, scratch, features, labels, learn_rate)
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GMM does not currently support backpropagation")


class _ApplyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, initial_labels, compiled_extention):
        return compiled_extention.gmm(features, initial_labels)
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GMM does not currently support backpropagation")
