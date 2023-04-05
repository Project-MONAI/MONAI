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
import torch.nn as nn
from typing import Optional

class HeadController(nn.Module):
    """
    Text-based controller for segmentation outputs, the text-driven segmentor enables for optional outputs instead of
    fixed output channels. Users can choose and control the number and name of output channels from a mult-class segmentation
    model. This can enabble incremental learning by adding new classes to a existing pre-trained model without 
    catatrophic forgetting.
    
    Text-dirven segmentor, based on: "Liu et al.,
    CLIP-Driven Universal Model for Organ Segmentation and Tumor Detection <https://arxiv.org/pdf/2301.00785.pdf>"
    """    
    def __init__(
        self,
        out_channels: int,
        weight_nums: list = [64, 64, 8],
        bias_nums: list = [8, 8, 1],
        hidden_size: int = 256,
        task_encoding: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            out_channels: number of output channels, to control text-baesd embedding for classes.
            weight_nums: weight feature size of text-driven segmentor conv layers, len(weight_nums) defines the number of layers.
            bias_nums: bias size of text-driven segmentor conv layers, len(bias_nums) needs to be consistent to len(weight_nums).
            hidden_size: dimension of hidden features, compatible to different vision feature dimensions.
            task_encoding: the text embedding features passed. TODO: make optional
        """
        super().__init__()
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums


        self.class_num = out_channels
        self.task_encoding = task_encoding
        if task_encoding:
            self.task_encoding = task_encoding
            self.controller = nn.Conv3d(2*hidden_size, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)
        else:
            self.controller = nn.Conv3d(hidden_size, sum(weight_nums+bias_nums), kernel_size=1, stride=1, padding=0)

        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 48),
            nn.ReLU(inplace=True),
            nn.Conv3d(48, 8, kernel_size=1)
        )

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        """
        Text-driven segmentor with layers of conv for dynamic outputs
        """
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 1)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = nn.functional.conv3d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = nn.functional.relu(x)
        return x

    def forward(self, x, out, logits_options=None):
        logits_options = range(x.shape[0]) if not isinstance(logits_options, list) else logits_options
        logits_array = []
        for i in logits_options:
            if self.task_encoding:
                x_cond = torch.cat([x[i].unsqueeze(0).repeat(self.class_num,1,1,1,1), self.task_encoding], 1)
            else:
                x_cond = x[i].unsqueeze(0).repeat(self.class_num,1,1,1,1)

            params = self.controller(x_cond)
            params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
            
            head_inputs = self.precls_conv(out[i].unsqueeze(0))
            head_inputs = head_inputs.repeat(self.class_num,1,1,1,1)
            N, _, D, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, D, H, W)
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)

            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits.reshape(1, -1, D, H, W))
        
        out = torch.cat(logits_array,dim=0)
        return out



