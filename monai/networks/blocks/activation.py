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
from torch import nn


class Swish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Swish}(x) = x * \text{Sigmoid}(\alpha * x) for constant value alpha.

    Citation: Searching for Activation Functions, Ramachandran et al., 2017, https://arxiv.org/abs/1710.05941.


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> m = Act['swish']()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(self.alpha * input)


class Mish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Mish}(x) = x * tanh(\text{softplus}(x)).

    Citation: Mish: A Self Regularized Non-Monotonic Activation Function, Diganta Misra, 2019, https://arxiv.org/abs/1908.08681.


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> m = Act['mish']()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.tanh(torch.nn.functional.softplus(input))
