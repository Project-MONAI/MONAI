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

import torch

from monai._extensions.loader import load_module

__all__ = ["GaussianMixtureModel"]


class GaussianMixtureModel:
    """
    Takes an initial labeling and uses a mixture of Gaussians to approximate each classes
    distribution in the feature space. Each unlabeled element is then assigned a probability
    of belonging to each class based on it's fit to each classes approximated distribution.

    See:
        https://en.wikipedia.org/wiki/Mixture_model
    """

    def __init__(self, channel_count: int, mixture_count: int, mixture_size: int, verbose_build: bool = False):
        """
        Args:
            channel_count: The number of features per element.
            mixture_count: The number of class distributions.
            mixture_size: The number Gaussian components per class distribution.
            verbose_build: If ``True``, turns on verbose logging of load steps.
        """
        if not torch.cuda.is_available():
            raise NotImplementedError("GaussianMixtureModel is currently implemented for CUDA.")
        self.channel_count = channel_count
        self.mixture_count = mixture_count
        self.mixture_size = mixture_size
        self.compiled_extension = load_module(
            "gmm",
            {"CHANNEL_COUNT": channel_count, "MIXTURE_COUNT": mixture_count, "MIXTURE_SIZE": mixture_size},
            verbose_build=verbose_build,
        )
        self.params, self.scratch = self.compiled_extension.init()

    def reset(self):
        """
        Resets the parameters of the model.
        """
        self.params, self.scratch = self.compiled_extension.init()

    def learn(self, features, labels):
        """
        Learns, from scratch, the distribution of each class from the provided labels.

        Args:
            features (torch.Tensor): features for each element.
            labels (torch.Tensor): initial labeling for each element.
        """
        self.compiled_extension.learn(self.params, self.scratch, features, labels)

    def apply(self, features):
        """
        Applies the current model to a set of feature vectors.

        Args:
            features (torch.Tensor): feature vectors for each element.

        Returns:
            output (torch.Tensor): class assignment probabilities for each element.
        """
        return _ApplyFunc.apply(self.params, features, self.compiled_extension)


class _ApplyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params, features, compiled_extension):
        return compiled_extension.apply(params, features)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("GMM does not support backpropagation")
