# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from torch import nn
from torch.autograd import Function
import torch

from monai.utils import optional_import

lltm_cpp, _ = optional_import("lltm_cpp")
lltm_cuda, _ = optional_import("lltm_cuda")


class LLTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        ext = lltm_cuda if weights.is_cuda else lltm_cpp
        outputs = ext.forward(input, weights, bias, old_h, old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights]
        ctx.save_for_backward(*variables)

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        if grad_h.is_cuda:
            outputs = lltm_cuda.backward(grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
            d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        else:
            outputs = lltm_cpp.backward(grad_h.contiguous(), grad_cell.contiguous(), *ctx.saved_tensors)
            d_old_h, d_input, d_weights, d_bias, d_old_cell = outputs

        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class LLTM(nn.Module):
    """
    This recurrent unit is similar to an LSTM, but differs in that it lacks a forget
    gate and uses an Exponential Linear Unit (ELU) as its internal activation function.
    Because this unit never forgets, call it LLTM, or Long-Long-Term-Memory unit.
    It has both C++ and CUDA implementation, automatically switch according to the
    target device where put this module to.

    Args:
        input_features: size of input feature data
        state_size: size of the state of recurrent unit

    Referring to: https://pytorch.org/tutorials/advanced/cpp_extension.html
    """

    def __init__(self, input_features: int, state_size: int):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        self.weights = nn.Parameter(torch.empty(3 * state_size, input_features + state_size))
        self.bias = nn.Parameter(torch.empty(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        return LLTMFunction.apply(input, self.weights, self.bias, *state)
