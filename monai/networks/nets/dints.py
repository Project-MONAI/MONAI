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


import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dints_block import (
    ActiConvNormBlock,
    FactorizedIncreaseBlock,
    FactorizedReduceBlock,
    P3DActiConvNormBlock,
)
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils import optional_import

# solving shortest path problem
csr_matrix, _ = optional_import("scipy.sparse", name="csr_matrix")
dijkstra, _ = optional_import("scipy.sparse.csgraph", name="dijkstra")

__all__ = ["DiNTS", "TopologyConstruction", "TopologyInstance", "TopologySearch"]


@torch.jit.interface
class CellInterface(torch.nn.Module):
    """interface for torchscriptable Cell"""

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        pass


@torch.jit.interface
class StemInterface(torch.nn.Module):
    """interface for torchscriptable Stem"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class StemTS(StemInterface):
    """wrapper for torchscriptable Stem"""

    def __init__(self, *mod):
        super().__init__()
        self.mod = torch.nn.Sequential(*mod)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mod(x)  # type: ignore


def _dfs(node, paths):
    """use depth first search to find all path activation combination"""
    if node == paths:
        return [[0], [1]]
    child = _dfs(node + 1, paths)
    return [[0] + _ for _ in child] + [[1] + _ for _ in child]


class _IdentityWithRAMCost(nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ram_cost = 0


class _CloseWithRAMCost(nn.Module):
    def __init__(self):
        super().__init__()
        self.ram_cost = 0

    def forward(self, x):
        return torch.tensor(0.0, requires_grad=False).to(x)


class _ActiConvNormBlockWithRAMCost(ActiConvNormBlock):
    """The class wraps monai layers with ram estimation. The ram_cost = total_ram/output_size is estimated.
    Here is the estimation:
     feature_size = output_size/out_channel
     total_ram = ram_cost * output_size
     total_ram = in_channel * feature_size (activation map) +
                 in_channel * feature_size (convolution map) +
                 out_channel * feature_size (normalization)
               = (2*in_channel + out_channel) * output_size/out_channel
     ram_cost = total_ram/output_size = 2 * in_channel/out_channel + 1
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__(in_channel, out_channel, kernel_size, padding, spatial_dims, act_name, norm_name)
        self.ram_cost = 1 + in_channel / out_channel * 2


class _P3DActiConvNormBlockWithRAMCost(P3DActiConvNormBlock):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        padding: int,
        p3dmode: int = 0,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__(in_channel, out_channel, kernel_size, padding, p3dmode, act_name, norm_name)
        # 1 in_channel (activation) + 1 in_channel (convolution) +
        # 1 out_channel (convolution) + 1 out_channel (normalization)
        self.ram_cost = 2 + 2 * in_channel / out_channel


class _FactorizedIncreaseBlockWithRAMCost(FactorizedIncreaseBlock):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__(in_channel, out_channel, spatial_dims, act_name, norm_name)
        # s0 is upsampled 2x from s1, representing feature sizes at two resolutions.
        # 2 * in_channel * s0 (upsample + activation) + 2 * out_channel * s0 (conv + normalization)
        # s0 = output_size/out_channel
        self.ram_cost = 2 * in_channel / out_channel + 2


class _FactorizedReduceBlockWithRAMCost(FactorizedReduceBlock):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__(in_channel, out_channel, spatial_dims, act_name, norm_name)
        # s0 is upsampled 2x from s1, representing feature sizes at two resolutions.
        # in_channel * s0 (activation) + 3 * out_channel * s1 (convolution, concatenation, normalization)
        # s0 = s1 * 2^(spatial_dims) = output_size / out_channel * 2^(spatial_dims)
        self.ram_cost = in_channel / out_channel * 2**self._spatial_dims + 3


class MixedOp(nn.Module):
    """
    The weighted averaging of cell operations.
    Args:
        c: number of output channels.
        ops: a dictionary of operations. See also: ``Cell.OPS2D`` or ``Cell.OPS3D``.
        arch_code_c: binary cell operation code. It represents the operation results added to the output.
    """

    def __init__(self, c: int, ops: dict, arch_code_c=None):
        super().__init__()
        if arch_code_c is None:
            arch_code_c = np.ones(len(ops))
        self.ops = nn.ModuleList()
        for arch_c, op_name in zip(arch_code_c, ops):
            self.ops.append(_CloseWithRAMCost() if arch_c == 0 else ops[op_name](c))

    def forward(self, x: torch.Tensor, weight: torch.Tensor):
        """
        Args:
            x: input tensor.
            weight: learnable architecture weights for cell operations. arch_code_c are derived from it.
        Return:
            out: weighted average of the operation results.
        """
        out = 0.0
        weight = weight.to(x)
        for idx, _op in enumerate(self.ops):
            out = out + _op(x) * weight[idx]
        return out


class Cell(CellInterface):
    """
    The basic class for cell operation search, which contains a preprocessing operation and a mixed cell operation.
    Each cell is defined on a `path` in the topology search space.
    Args:
        c_prev: number of input channels
        c: number of output channels
        rate: resolution change rate. It represents the preprocessing operation before the mixed cell operation.
            ``-1`` for 2x downsample, ``1`` for 2x upsample, ``0`` for no change of resolution.
        arch_code_c: cell operation code
    """

    DIRECTIONS = 3
    # Possible output paths for `Cell`.
    #
    #       - UpSample
    #      /
    # +--+/
    # |  |--- Identity or AlignChannels
    # +--+\
    #      \
    #       - Downsample

    # Define 2D operation set, parameterized by the number of channels
    OPS2D = {
        "skip_connect": lambda _c: _IdentityWithRAMCost(),
        "conv_3x3": lambda c: _ActiConvNormBlockWithRAMCost(c, c, 3, padding=1, spatial_dims=2),
    }

    # Define 3D operation set, parameterized by the number of channels
    OPS3D = {
        "skip_connect": lambda _c: _IdentityWithRAMCost(),
        "conv_3x3x3": lambda c: _ActiConvNormBlockWithRAMCost(c, c, 3, padding=1, spatial_dims=3),
        "conv_3x3x1": lambda c: _P3DActiConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=0),
        "conv_3x1x3": lambda c: _P3DActiConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=1),
        "conv_1x3x3": lambda c: _P3DActiConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=2),
    }

    # Define connection operation set, parameterized by the number of channels
    ConnOPS = {
        "up": _FactorizedIncreaseBlockWithRAMCost,
        "down": _FactorizedReduceBlockWithRAMCost,
        "identity": _IdentityWithRAMCost,
        "align_channels": _ActiConvNormBlockWithRAMCost,
    }

    def __init__(
        self,
        c_prev: int,
        c: int,
        rate: int,
        arch_code_c=None,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
    ):
        super().__init__()
        self._spatial_dims = spatial_dims
        self._act_name = act_name
        self._norm_name = norm_name

        if rate == -1:  # downsample
            self.preprocess = self.ConnOPS["down"](
                c_prev, c, spatial_dims=self._spatial_dims, act_name=self._act_name, norm_name=self._norm_name
            )
        elif rate == 1:  # upsample
            self.preprocess = self.ConnOPS["up"](
                c_prev, c, spatial_dims=self._spatial_dims, act_name=self._act_name, norm_name=self._norm_name
            )
        else:
            if c_prev == c:
                self.preprocess = self.ConnOPS["identity"]()
            else:
                self.preprocess = self.ConnOPS["align_channels"](
                    c_prev, c, 1, 0, spatial_dims=self._spatial_dims, act_name=self._act_name, norm_name=self._norm_name
                )

        # Define 2D operation set, parameterized by the number of channels
        self.OPS2D = {
            "skip_connect": lambda _c: _IdentityWithRAMCost(),
            "conv_3x3": lambda c: _ActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, spatial_dims=2, act_name=self._act_name, norm_name=self._norm_name
            ),
        }

        # Define 3D operation set, parameterized by the number of channels
        self.OPS3D = {
            "skip_connect": lambda _c: _IdentityWithRAMCost(),
            "conv_3x3x3": lambda c: _ActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, spatial_dims=3, act_name=self._act_name, norm_name=self._norm_name
            ),
            "conv_3x3x1": lambda c: _P3DActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, p3dmode=0, act_name=self._act_name, norm_name=self._norm_name
            ),
            "conv_3x1x3": lambda c: _P3DActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, p3dmode=1, act_name=self._act_name, norm_name=self._norm_name
            ),
            "conv_1x3x3": lambda c: _P3DActiConvNormBlockWithRAMCost(
                c, c, 3, padding=1, p3dmode=2, act_name=self._act_name, norm_name=self._norm_name
            ),
        }

        self.OPS = {}
        if self._spatial_dims == 2:
            self.OPS = self.OPS2D
        elif self._spatial_dims == 3:
            self.OPS = self.OPS3D
        else:
            raise NotImplementedError(f"Spatial dimensions {self._spatial_dims} is not supported.")

        self.op = MixedOp(c, self.OPS, arch_code_c)

    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
            weight: weights for different operations.
        """
        x = self.preprocess(x)
        x = self.op(x, weight)
        return x


class DiNTS(nn.Module):
    """
    Reimplementation of DiNTS based on
    "DiNTS: Differentiable Neural Network Topology Search for 3D Medical Image Segmentation
    <https://arxiv.org/abs/2103.15954>".

    The model contains a pre-defined multi-resolution stem block (defined in this class) and a
    DiNTS space (defined in :py:class:`monai.networks.nets.TopologyInstance` and
    :py:class:`monai.networks.nets.TopologySearch`).

    The stem block is for: 1) input downsample and 2) output upsample to original size.
    The model downsamples the input image by 2 (if ``use_downsample=True``).
    The downsampled image is downsampled by [1, 2, 4, 8] times (``num_depths=4``) and used as input to the
    DiNTS search space (``TopologySearch``) or the DiNTS instance (``TopologyInstance``).

        - ``TopologyInstance`` is the final searched model. The initialization requires the searched architecture codes.
        - ``TopologySearch`` is a multi-path topology and cell operation search space.
          The architecture codes will be initialized as one.
        - ``TopologyConstruction`` is the parent class which constructs the instance and search space.

    To meet the requirements of the structure, the input size for each spatial dimension should be:
    divisible by 2 ** (num_depths + 1).

    Args:
        dints_space: DiNTS search space. The value should be instance of `TopologyInstance` or `TopologySearch`.
        in_channels: number of input image channels.
        num_classes: number of output segmentation classes.
        act_name: activation name, default to 'RELU'.
        norm_name: normalization used in convolution blocks. Default to `InstanceNorm`.
        spatial_dims: spatial 2D or 3D inputs.
        use_downsample: use downsample in the stem.
            If ``False``, the search space will be in resolution [1, 1/2, 1/4, 1/8],
            if ``True``, the search space will be in resolution [1/2, 1/4, 1/8, 1/16].
        node_a: node activation numpy matrix. Its shape is `(num_depths, num_blocks + 1)`.
            +1 for multi-resolution inputs.
            In model searching stage, ``node_a`` can be None. In deployment stage, ``node_a`` cannot be None.
    """

    def __init__(
        self,
        dints_space,
        in_channels: int,
        num_classes: int,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        spatial_dims: int = 3,
        use_downsample: bool = True,
        node_a=None,
    ):
        super().__init__()

        self.dints_space = dints_space
        self.filter_nums = dints_space.filter_nums
        self.num_blocks = dints_space.num_blocks
        self.num_depths = dints_space.num_depths
        if spatial_dims not in (2, 3):
            raise NotImplementedError(f"Spatial dimensions {spatial_dims} is not supported.")
        self._spatial_dims = spatial_dims
        if node_a is None:
            self.node_a = torch.ones((self.num_blocks + 1, self.num_depths))
        else:
            self.node_a = node_a

        # define stem operations for every block
        conv_type = Conv[Conv.CONV, spatial_dims]
        self.stem_down = nn.ModuleDict()
        self.stem_up = nn.ModuleDict()
        self.stem_finals = nn.Sequential(
            ActiConvNormBlock(
                self.filter_nums[0],
                self.filter_nums[0],
                act_name=act_name,
                norm_name=norm_name,
                spatial_dims=spatial_dims,
            ),
            conv_type(
                in_channels=self.filter_nums[0],
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=True,
                dilation=1,
            ),
        )
        mode = "trilinear" if self._spatial_dims == 3 else "bilinear"
        for res_idx in range(self.num_depths):
            # define downsample stems before DiNTS search
            if use_downsample:
                self.stem_down[str(res_idx)] = StemTS(
                    nn.Upsample(scale_factor=1 / (2**res_idx), mode=mode, align_corners=True),
                    conv_type(
                        in_channels=in_channels,
                        out_channels=self.filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[res_idx]),
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=self.filter_nums[res_idx],
                        out_channels=self.filter_nums[res_idx + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[res_idx + 1]),
                )
                self.stem_up[str(res_idx)] = StemTS(
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=self.filter_nums[res_idx + 1],
                        out_channels=self.filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[res_idx]),
                    nn.Upsample(scale_factor=2, mode=mode, align_corners=True),
                )

            else:
                self.stem_down[str(res_idx)] = StemTS(
                    nn.Upsample(scale_factor=1 / (2**res_idx), mode=mode, align_corners=True),
                    conv_type(
                        in_channels=in_channels,
                        out_channels=self.filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[res_idx]),
                )
                self.stem_up[str(res_idx)] = StemTS(
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=self.filter_nums[res_idx],
                        out_channels=self.filter_nums[max(res_idx - 1, 0)],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(
                        name=norm_name, spatial_dims=spatial_dims, channels=self.filter_nums[max(res_idx - 1, 0)]
                    ),
                    nn.Upsample(scale_factor=2 ** (res_idx != 0), mode=mode, align_corners=True),
                )

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]

    def forward(self, x: torch.Tensor):
        """
        Prediction based on dynamic arch_code.

        Args:
            x: input tensor.
        """
        inputs = []
        for d in range(self.num_depths):
            # allow multi-resolution input
            _mod_w: StemInterface = self.stem_down[str(d)]
            x_out = _mod_w.forward(x)
            if self.node_a[0][d]:
                inputs.append(x_out)
            else:
                inputs.append(torch.zeros_like(x_out))

        outputs = self.dints_space(inputs)

        blk_idx = self.num_blocks - 1
        start = False
        _temp: torch.Tensor = torch.empty(0)
        for res_idx in range(self.num_depths - 1, -1, -1):
            _mod_up: StemInterface = self.stem_up[str(res_idx)]
            if start:
                _temp = _mod_up.forward(outputs[res_idx] + _temp)
            elif self.node_a[blk_idx + 1][res_idx]:
                start = True
                _temp = _mod_up.forward(outputs[res_idx])
        prediction = self.stem_finals(_temp)
        return prediction


class TopologyConstruction(nn.Module):
    """
    The base class for `TopologyInstance` and `TopologySearch`.

    Args:
        arch_code: `[arch_code_a, arch_code_c]`, numpy arrays. The architecture codes defining the model.
            For example, for a ``num_depths=4, num_blocks=12`` search space:

            - `arch_code_a` is a 12x10 (10 paths) binary matrix representing if a path is activated.
            - `arch_code_c` is a 12x10x5 (5 operations) binary matrix representing if a cell operation is used.
            - `arch_code` in ``__init__()`` is used for creating the network and remove unused network blocks. If None,

            all paths and cells operations will be used, and must be in the searching stage (is_search=True).
        channel_mul: adjust intermediate channel number, default is 1.
        cell: operation of each node.
        num_blocks: number of blocks (depth in the horizontal direction) of the DiNTS search space.
        num_depths: number of image resolutions of the DiNTS search space: 1, 1/2, 1/4 ... in each dimension.
        use_downsample: use downsample in the stem. If False, the search space will be in resolution [1, 1/2, 1/4, 1/8],
            if True, the search space will be in resolution [1/2, 1/4, 1/8, 1/16].
        device: `'cpu'`, `'cuda'`, or device ID.


    Predefined variables:
        `filter_nums`: default to 32. Double the number of channels after downsample.
        topology related variables:

            - `arch_code2in`: path activation to its incoming node index (resolution). For depth = 4,
              arch_code2in = [0, 1, 0, 1, 2, 1, 2, 3, 2, 3]. The first path outputs from node 0 (top resolution),
              the second path outputs from node 1 (second resolution in the search space),
              the third path outputs from node 0, etc.
            - `arch_code2ops`: path activation to operations of upsample 1, keep 0, downsample -1. For depth = 4,
              arch_code2ops = [0, 1, -1, 0, 1, -1, 0, 1, -1, 0]. The first path does not change
              resolution, the second path perform upsample, the third perform downsample, etc.
            - `arch_code2out`: path activation to its output node index.
              For depth = 4, arch_code2out = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
              the first and second paths connects to node 0 (top resolution), the 3,4,5 paths connects to node 1, etc.
    """

    def __init__(
        self,
        arch_code: Optional[list] = None,
        channel_mul: float = 1.0,
        cell=Cell,
        num_blocks: int = 6,
        num_depths: int = 3,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        use_downsample: bool = True,
        device: str = "cpu",
    ):

        super().__init__()

        self.filter_nums = [int(n_feat * channel_mul) for n_feat in (32, 64, 128, 256, 512)]
        self.num_blocks = num_blocks
        self.num_depths = num_depths
        self._spatial_dims = spatial_dims
        self._act_name = act_name
        self._norm_name = norm_name
        self.use_downsample = use_downsample
        self.device = device
        self.num_cell_ops = 0
        if self._spatial_dims == 2:
            self.num_cell_ops = len(cell.OPS2D)
        elif self._spatial_dims == 3:
            self.num_cell_ops = len(cell.OPS3D)

        # Calculate predefined parameters for topology search and decoding
        arch_code2in, arch_code2out = [], []
        for i in range(Cell.DIRECTIONS * self.num_depths - 2):
            arch_code2in.append((i + 1) // Cell.DIRECTIONS - 1 + (i + 1) % Cell.DIRECTIONS)
        arch_code2ops = ([-1, 0, 1] * self.num_depths)[1:-1]
        for m in range(self.num_depths):
            arch_code2out.extend([m, m, m])
        arch_code2out = arch_code2out[1:-1]
        self.arch_code2in = arch_code2in
        self.arch_code2ops = arch_code2ops
        self.arch_code2out = arch_code2out

        # define NAS search space
        if arch_code is None:
            arch_code_a = torch.ones((self.num_blocks, len(self.arch_code2out))).to(self.device)
            arch_code_c = torch.ones((self.num_blocks, len(self.arch_code2out), self.num_cell_ops)).to(self.device)
        else:
            arch_code_a = torch.from_numpy(arch_code[0]).to(self.device)
            arch_code_c = F.one_hot(torch.from_numpy(arch_code[1]).to(torch.int64), self.num_cell_ops).to(self.device)

        self.arch_code_a = arch_code_a
        self.arch_code_c = arch_code_c
        # define cell operation on each path
        self.cell_tree = nn.ModuleDict()
        for blk_idx in range(self.num_blocks):
            for res_idx in range(len(self.arch_code2out)):
                if self.arch_code_a[blk_idx, res_idx] == 1:
                    self.cell_tree[str((blk_idx, res_idx))] = cell(
                        self.filter_nums[self.arch_code2in[res_idx] + int(use_downsample)],
                        self.filter_nums[self.arch_code2out[res_idx] + int(use_downsample)],
                        self.arch_code2ops[res_idx],
                        self.arch_code_c[blk_idx, res_idx],
                        self._spatial_dims,
                        self._act_name,
                        self._norm_name,
                    )

    def forward(self, x):
        """This function to be implemented by the architecture instances or search spaces."""
        pass


class TopologyInstance(TopologyConstruction):
    """
    Instance of the final searched architecture. Only used in re-training/inference stage.
    """

    def __init__(
        self,
        arch_code=None,
        channel_mul: float = 1.0,
        cell=Cell,
        num_blocks: int = 6,
        num_depths: int = 3,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        use_downsample: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize DiNTS topology search space of neural architectures.
        """
        if arch_code is None:
            warnings.warn("arch_code not provided when not searching.")

        super().__init__(
            arch_code=arch_code,
            channel_mul=channel_mul,
            cell=cell,
            num_blocks=num_blocks,
            num_depths=num_depths,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            use_downsample=use_downsample,
            device=device,
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            x: input tensor.
        """
        # generate path activation probability
        inputs, outputs = x, [torch.tensor(0.0).to(x[0])] * self.num_depths
        for blk_idx in range(self.num_blocks):
            outputs = [torch.tensor(0.0).to(x[0])] * self.num_depths
            for res_idx, activation in enumerate(self.arch_code_a[blk_idx].data):
                if activation:
                    mod: CellInterface = self.cell_tree[str((blk_idx, res_idx))]
                    _out = mod.forward(
                        x=inputs[self.arch_code2in[res_idx]], weight=torch.ones_like(self.arch_code_c[blk_idx, res_idx])
                    )
                    outputs[self.arch_code2out[res_idx]] = outputs[self.arch_code2out[res_idx]] + _out
            inputs = outputs

        return inputs


class TopologySearch(TopologyConstruction):
    """
    DiNTS topology search space of neural architectures.

    Examples:

    .. code-block:: python

        from monai.networks.nets.dints import TopologySearch

        topology_search_space = TopologySearch(
            channel_mul=0.5, num_blocks=8, num_depths=4, use_downsample=True, spatial_dims=3)
        topology_search_space.get_ram_cost_usage(in_size=(2, 16, 80, 80, 80), full=True)
        multi_res_images = [
            torch.randn(2, 16, 80, 80, 80),
            torch.randn(2, 32, 40, 40, 40),
            torch.randn(2, 64, 20, 20, 20),
            torch.randn(2, 128, 10, 10, 10)]
        prediction = topology_search_space(image)
        for x in prediction: print(x.shape)
        # torch.Size([2, 16, 80, 80, 80])
        # torch.Size([2, 32, 40, 40, 40])
        # torch.Size([2, 64, 20, 20, 20])
        # torch.Size([2, 128, 10, 10, 10])

    Class method overview:

        - ``get_prob_a()``: convert learnable architecture weights to path activation probabilities.
        - ``get_ram_cost_usage()``: get estimated ram cost.
        - ``get_topology_entropy()``: get topology entropy loss in searching stage.
        - ``decode()``: get final binarized architecture code.
        - ``gen_mtx()``: generate variables needed for topology search.

    Predefined variables:
        - `tidx`: index used to convert path activation matrix T = (depth,depth) in transfer_mtx to
          path activation arch_code (1,3*depth-2), for depth = 4, tidx = [0, 1, 4, 5, 6, 9, 10, 11, 14, 15],
          A tidx (10 binary values) represents the path activation.
        - `transfer_mtx`: feasible path activation matrix (denoted as T) given a node activation pattern.
          It is used to convert path activation pattern (1, paths) to node activation (1, nodes)
        - `node_act_list`: all node activation [2^num_depths-1, depth]. For depth = 4, there are 15 node activation
          patterns, each of length 4. For example, [1,1,0,0] means nodes 0, 1 are activated (with input paths).
        - `all_connect`: All possible path activations. For depth = 4,
          all_connection has 1024 vectors of length 10 (10 paths).
          The return value will exclude path activation of all 0.
    """

    def __init__(
        self,
        channel_mul: float = 1.0,
        cell=Cell,
        arch_code: Optional[list] = None,
        num_blocks: int = 6,
        num_depths: int = 3,
        spatial_dims: int = 3,
        act_name: Union[Tuple, str] = "RELU",
        norm_name: Union[Tuple, str] = ("INSTANCE", {"affine": True}),
        use_downsample: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize DiNTS topology search space of neural architectures.
        """
        super().__init__(
            arch_code=arch_code,
            channel_mul=channel_mul,
            cell=cell,
            num_blocks=num_blocks,
            num_depths=num_depths,
            spatial_dims=spatial_dims,
            act_name=act_name,
            norm_name=norm_name,
            use_downsample=use_downsample,
            device=device,
        )

        tidx = []
        _d = Cell.DIRECTIONS
        for i in range(_d * self.num_depths - 2):
            tidx.append((i + 1) // _d * self.num_depths + (i + 1) // _d - 1 + (i + 1) % _d)
        self.tidx = tidx
        transfer_mtx, node_act_list, child_list = self.gen_mtx(num_depths)

        self.node_act_list = np.asarray(node_act_list)
        self.node_act_dict = {str(self.node_act_list[i]): i for i in range(len(self.node_act_list))}
        self.transfer_mtx = transfer_mtx
        self.child_list = np.asarray(child_list)

        self.ram_cost = np.zeros((self.num_blocks, len(self.arch_code2out), self.num_cell_ops))
        for blk_idx in range(self.num_blocks):
            for res_idx in range(len(self.arch_code2out)):
                if self.arch_code_a[blk_idx, res_idx] == 1:
                    self.ram_cost[blk_idx, res_idx] = np.array(
                        [
                            op.ram_cost + self.cell_tree[str((blk_idx, res_idx))].preprocess.ram_cost
                            for op in self.cell_tree[str((blk_idx, res_idx))].op.ops[: self.num_cell_ops]
                        ]
                    )

        # define cell and macro architecture probabilities
        self.log_alpha_c = nn.Parameter(
            torch.zeros(self.num_blocks, len(self.arch_code2out), self.num_cell_ops)
            .normal_(1, 0.01)
            .to(self.device)
            .requires_grad_()
        )
        self.log_alpha_a = nn.Parameter(
            torch.zeros(self.num_blocks, len(self.arch_code2out)).normal_(0, 0.01).to(self.device).requires_grad_()
        )
        self._arch_param_names = ["log_alpha_a", "log_alpha_c"]

    def gen_mtx(self, depth: int):
        """
        Generate elements needed in decoding and topology.

            - `transfer_mtx`: feasible path activation matrix (denoted as T) given a node activation pattern.
               It is used to convert path activation pattern (1, paths) to node activation (1, nodes)
            - `node_act_list`: all node activation [2^num_depths-1, depth]. For depth = 4, there are 15 node activation
               patterns, each of length 4. For example, [1,1,0,0] means nodes 0, 1 are activated (with input paths).
            - `all_connect`: All possible path activations. For depth = 4,
              all_connection has 1024 vectors of length 10 (10 paths).
              The return value will exclude path activation of all 0.
        """
        # total paths in a block, each node has three output paths,
        # except the two nodes at the top and the bottom scales
        paths = Cell.DIRECTIONS * depth - 2

        # for 10 paths, all_connect has 1024 possible path activations. [1 0 0 0 0 0 0 0 0 0] means the top
        # path is activated.
        all_connect = _dfs(0, paths - 1)

        # Save all possible connections in mtx (might be redundant and infeasible)
        mtx = []
        for m in all_connect:
            # convert path activation [1,paths] to path activation matrix [depth, depth]
            ma = np.zeros((depth, depth))
            for i in range(paths):
                ma[(i + 1) // Cell.DIRECTIONS, (i + 1) // Cell.DIRECTIONS - 1 + (i + 1) % Cell.DIRECTIONS] = m[i]
            mtx.append(ma)

        # define all possible node activation
        node_act_list = _dfs(0, depth - 1)[1:]
        transfer_mtx = {}
        for arch_code in node_act_list:
            # make sure each activated node has an active connection, inactivated node has no connection
            arch_code_mtx = [_ for _ in mtx if ((np.sum(_, 0) > 0).astype(int) == np.array(arch_code)).all()]
            transfer_mtx[str(np.array(arch_code))] = arch_code_mtx

        return transfer_mtx, node_act_list, all_connect[1:]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def get_prob_a(self, child: bool = False):
        """
        Get final path and child model probabilities from architecture weights `log_alpha_a`.
        This is used in forward pass, getting training loss, and final decoding.

        Args:
            child: return child probability (used in decoding)
        Return:
            arch_code_prob_a: the path activation probability of size:
                `[number of blocks, number of paths in each block]`.
                For 12 blocks, 4 depths search space, the size is [12,10]
            probs_a: The probability of all child models (size 1023x10). Each child model is a path activation pattern
                 (1D vector of length 10 for 10 paths). In total 1023 child models (2^10 -1)
        """
        _arch_code_prob_a = torch.sigmoid(self.log_alpha_a)
        # remove the case where all path are zero, and re-normalize.
        norm = 1 - (1 - _arch_code_prob_a).prod(-1)
        arch_code_prob_a = _arch_code_prob_a / norm.unsqueeze(1)
        if child:
            path_activation = torch.from_numpy(self.child_list).to(self.device)
            probs_a = [
                (
                    path_activation * _arch_code_prob_a[blk_idx]
                    + (1 - path_activation) * (1 - _arch_code_prob_a[blk_idx])
                ).prod(-1)
                / norm[blk_idx]
                for blk_idx in range(self.num_blocks)
            ]
            probs_a = torch.stack(probs_a)  # type: ignore
            return probs_a, arch_code_prob_a
        return None, arch_code_prob_a

    def get_ram_cost_usage(self, in_size, full: bool = False):
        """
        Get estimated output tensor size to approximate RAM consumption.

        Args:
            in_size: input image shape (4D/5D, ``[BCHW[D]]``) at the highest resolution level.
            full: full ram cost usage with all probability of 1.
        """
        # convert input image size to feature map size at each level
        batch_size = in_size[0]
        image_size = np.array(in_size[-self._spatial_dims :])
        sizes = []
        for res_idx in range(self.num_depths):
            sizes.append(batch_size * self.filter_nums[res_idx] * (image_size // (2**res_idx)).prod())
        sizes = torch.tensor(sizes).to(torch.float32).to(self.device) / (2 ** (int(self.use_downsample)))
        probs_a, arch_code_prob_a = self.get_prob_a(child=False)
        cell_prob = F.softmax(self.log_alpha_c, dim=-1)
        if full:
            arch_code_prob_a = arch_code_prob_a.detach()
            arch_code_prob_a.fill_(1)
        ram_cost = torch.from_numpy(self.ram_cost).to(torch.float32).to(self.device)
        usage = 0.0
        for blk_idx in range(self.num_blocks):
            # node activation for input
            # cell operation
            for path_idx in range(len(self.arch_code2out)):
                usage += (
                    arch_code_prob_a[blk_idx, path_idx]
                    * (1 + (ram_cost[blk_idx, path_idx] * cell_prob[blk_idx, path_idx]).sum())
                    * sizes[self.arch_code2out[path_idx]]
                )
        return usage * 32 / 8 / 1024**2

    def get_topology_entropy(self, probs):
        """
        Get topology entropy loss at searching stage.

        Args:
            probs: path activation probabilities
        """
        if hasattr(self, "node2in"):
            node2in = self.node2in  # pylint: disable=E0203
            node2out = self.node2out  # pylint: disable=E0203
        else:
            # node activation index to feasible input child_idx
            node2in = [[] for _ in range(len(self.node_act_list))]
            # node activation index to feasible output child_idx
            node2out = [[] for _ in range(len(self.node_act_list))]
            for child_idx in range(len(self.child_list)):
                _node_in, _node_out = np.zeros(self.num_depths), np.zeros(self.num_depths)
                for res_idx in range(len(self.arch_code2out)):
                    _node_out[self.arch_code2out[res_idx]] += self.child_list[child_idx][res_idx]
                    _node_in[self.arch_code2in[res_idx]] += self.child_list[child_idx][res_idx]
                _node_in = (_node_in >= 1).astype(int)
                _node_out = (_node_out >= 1).astype(int)
                node2in[self.node_act_dict[str(_node_out)]].append(child_idx)
                node2out[self.node_act_dict[str(_node_in)]].append(child_idx)
            self.node2in = node2in
            self.node2out = node2out
        # calculate entropy
        ent = 0
        for blk_idx in range(self.num_blocks - 1):
            blk_ent = 0
            # node activation probability
            for node_idx in range(len(self.node_act_list)):
                _node_p = probs[blk_idx, node2in[node_idx]].sum()
                _out_probs = probs[blk_idx + 1, node2out[node_idx]].sum()
                blk_ent += -(_node_p * torch.log(_out_probs + 1e-5) + (1 - _node_p) * torch.log(1 - _out_probs + 1e-5))
            ent += blk_ent
        return ent

    def decode(self):
        """
        Decode network log_alpha_a/log_alpha_c using dijkstra shortest path algorithm.

        `[node_a, arch_code_a, arch_code_c, arch_code_a_max]` is decoded when using ``self.decode()``.

        For example, for a ``num_depths=4``, ``num_blocks=12`` search space:

            - ``node_a`` is a 4x13 binary matrix representing if a feature node is activated
              (13 because of multi-resolution inputs).
            - ``arch_code_a`` is a 12x10 (10 paths) binary matrix representing if a path is activated.
            - ``arch_code_c`` is a 12x10x5 (5 operations) binary matrix representing if a cell operation is used.

        Return:
            arch_code with maximum probability
        """
        probs, arch_code_prob_a = self.get_prob_a(child=True)
        arch_code_a_max = self.child_list[torch.argmax(probs, -1).data.cpu().numpy()]
        arch_code_c = torch.argmax(F.softmax(self.log_alpha_c, -1), -1).data.cpu().numpy()
        probs = probs.data.cpu().numpy()

        # define adjacency matrix
        amtx = np.zeros(
            (1 + len(self.child_list) * self.num_blocks + 1, 1 + len(self.child_list) * self.num_blocks + 1)
        )

        # build a path activation to child index searching dictionary
        path2child = {str(self.child_list[i]): i for i in range(len(self.child_list))}

        # build a submodel to submodel index
        sub_amtx = np.zeros((len(self.child_list), len(self.child_list)))
        for child_idx in range(len(self.child_list)):
            _node_act = np.zeros(self.num_depths).astype(int)
            for path_idx in range(len(self.child_list[child_idx])):
                _node_act[self.arch_code2out[path_idx]] += self.child_list[child_idx][path_idx]
            _node_act = (_node_act >= 1).astype(int)
            for mtx in self.transfer_mtx[str(_node_act)]:
                connect_child_idx = path2child[str(mtx.flatten()[self.tidx].astype(int))]
                sub_amtx[child_idx, connect_child_idx] = 1

        # fill in source to first block, add 1e-5/1e-3 to avoid log0 and negative edge weights
        amtx[0, 1 : 1 + len(self.child_list)] = -np.log(probs[0] + 1e-5) + 0.001

        # fill in the rest blocks
        for blk_idx in range(1, self.num_blocks):
            amtx[
                1 + (blk_idx - 1) * len(self.child_list) : 1 + blk_idx * len(self.child_list),
                1 + blk_idx * len(self.child_list) : 1 + (blk_idx + 1) * len(self.child_list),
            ] = sub_amtx * np.tile(-np.log(probs[blk_idx] + 1e-5) + 0.001, (len(self.child_list), 1))

        # fill in the last to the sink
        amtx[1 + (self.num_blocks - 1) * len(self.child_list) : 1 + self.num_blocks * len(self.child_list), -1] = 0.001

        graph = csr_matrix(amtx)
        dist_matrix, predecessors, sources = dijkstra(
            csgraph=graph, directed=True, indices=0, min_only=True, return_predecessors=True
        )
        index, a_idx = -1, -1
        arch_code_a = np.zeros((self.num_blocks, len(self.arch_code2out)))
        node_a = np.zeros((self.num_blocks + 1, self.num_depths))

        # decoding to paths
        while True:
            index = predecessors[index]
            if index == 0:
                break
            child_idx = (index - 1) % len(self.child_list)
            arch_code_a[a_idx, :] = self.child_list[child_idx]
            for res_idx in range(len(self.arch_code2out)):
                node_a[a_idx, self.arch_code2out[res_idx]] += arch_code_a[a_idx, res_idx]
            a_idx -= 1
        for res_idx in range(len(self.arch_code2out)):
            node_a[a_idx, self.arch_code2in[res_idx]] += arch_code_a[0, res_idx]
        node_a = (node_a >= 1).astype(int)
        return node_a, arch_code_a, arch_code_c, arch_code_a_max

    def forward(self, x):
        """
        Prediction based on dynamic arch_code.

        Args:
            x: a list of `num_depths` input tensors as a multi-resolution input.
                tensor is of shape `BCHW[D]` where `C` must match `self.filter_nums`.
        """
        # generate path activation probability
        probs_a, arch_code_prob_a = self.get_prob_a(child=False)
        inputs = x
        for blk_idx in range(self.num_blocks):
            outputs = [0.0] * self.num_depths
            for res_idx, activation in enumerate(self.arch_code_a[blk_idx].data.cpu().numpy()):
                if activation:
                    _w = F.softmax(self.log_alpha_c[blk_idx, res_idx], dim=-1)
                    outputs[self.arch_code2out[res_idx]] += (
                        self.cell_tree[str((blk_idx, res_idx))](inputs[self.arch_code2in[res_idx]], weight=_w)
                        * arch_code_prob_a[blk_idx, res_idx]
                    )
            inputs = outputs

        return inputs
