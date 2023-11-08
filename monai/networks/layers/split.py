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

import os

import monai
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence
from monai.networks.blocks import Convolution


NUM_SPLITS = 16
SPLIT_PADDING = 3

class SplitConvolution(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NDA",
        act: tuple | str | None = "PRELU",
        norm: tuple | str | None = "INSTANCE",
        dropout: tuple | str | float | None = None,
        dropout_dim: int | None = 1,
        dilation: Sequence[int] | int = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Sequence[int] | int | None = None,
        output_padding: Sequence[int] | int | None = None,
    ) -> None:
        super(SplitConvolution, self).__init__()
        self.conv = monai.networks.blocks.convolutions.Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            adn_ordering,
            act,
            norm,
            dropout,
            dropout_dim,
            dilation,
            groups,
            bias,
            conv_only,
            is_transposed,
            padding,
            output_padding,
        )

        self.tp_dim = 1

    def forward(self, x):        
        num_splits = NUM_SPLITS
        print("num_splits:", num_splits)
        l = x.size(self.tp_dim + 2)
        split_size = l // num_splits

        if False:
            splits = [x[:, :, i * split_size : (i+1) * split_size, :, :] for i in range(num_splits)]
        else:
            padding = SPLIT_PADDING
            print("padding:", padding)

            overlaps = [0] + [padding] * (num_splits - 1)
            if self.tp_dim == 0:
                splits = [x[:, :, i * split_size - overlaps[i] : (i+1) * split_size + (padding if i != num_splits - 1 else 0), :, :] for i in range(num_splits)]
            elif self.tp_dim == 1:
                splits = [x[:, :, :, i * split_size - overlaps[i] : (i+1) * split_size + (padding if i != num_splits - 1 else 0), :] for i in range(num_splits)]
            elif self.tp_dim == 2:
                splits = [x[:, :, :, :, i * split_size - overlaps[i] : (i+1) * split_size + (padding if i != num_splits - 1 else 0)] for i in range(num_splits)]

            for _j in range(len(splits)):
                print(f"splits {_j + 1}/{len(splits)}:", splits[_j].size())

        del x
        torch.cuda.empty_cache()

        splits_0_size = list(splits[0].size())
    
        if False:
            outputs = [self.conv(splits[i]) for i in range(num_splits)]
        else:
            outputs = []
            _type = splits[0].device.type
            for _i in range(num_splits):
                if _type == 'cuda':
                    outputs.append(self.conv(splits[_i]))
                else:
                    _t = splits[_i]
                    _t1 = self.conv(_t.to('cuda', non_blocking=True))
                    del _t
                    torch.cuda.empty_cache()
                    _t1 = _t1.to('cpu', non_blocking=True)
                    outputs.append(_t1)
                    del _t1
                    torch.cuda.empty_cache()

                splits[_i] = 0
                torch.cuda.empty_cache()

        del splits
        torch.cuda.empty_cache()

        split_size_out = split_size
        padding_s = padding
        non_tp_dim = self.tp_dim + 1 if self.tp_dim < 2 else 0
        if outputs[0].size(non_tp_dim + 2) // splits_0_size[non_tp_dim + 2] == 2:
            split_size_out *= 2
            padding_s *= 2
        elif splits_0_size[non_tp_dim + 2] // outputs[0].size(non_tp_dim + 2) == 2:
            split_size_out = split_size_out // 2
            padding_s = padding_s // 2

        if self.tp_dim == 0:
            outputs[0] = outputs[0][:, :, :split_size_out, :, :]
            for i in range(1, num_splits):
                outputs[i] = outputs[i][:, :, padding_s:padding_s + split_size_out, :, :]
        elif self.tp_dim == 1:
            outputs[0] = outputs[0][:, :, :, :split_size_out, :]
            for i in range(1, num_splits):
                outputs[i] = outputs[i][:, :, :, padding_s:padding_s + split_size_out, :]
        elif self.tp_dim == 2:
            outputs[0] = outputs[0][:, :, :, :, :split_size_out]
            for i in range(1, num_splits):
                outputs[i] = outputs[i][:, :, :, :, padding_s:padding_s + split_size_out]

        if max(outputs[0].size()) < 500:
            x = torch.cat([out for out in outputs], dim=self.tp_dim + 2)
        else:
            import gc

            _type = outputs[0].device.type
            if _type == 'cuda':
                x = outputs[0].clone().to('cpu', non_blocking=True)
            outputs[0] = 0
            torch.cuda.empty_cache()
            for _k in range(len(outputs) - 1):
                x = torch.cat((x, outputs[_k + 1].cpu()), dim=self.tp_dim + 2)
                outputs[_k + 1] = 0
                torch.cuda.empty_cache()
                gc.collect()
            if _type == 'cuda':
                x = x.to('cuda', non_blocking=True)

        del outputs
        torch.cuda.empty_cache()

        return x