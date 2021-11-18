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

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dints_block import (
    FactorizedIncreaseBlock,
    FactorizedReduceBlock,
    P3DReLUConvNormBlock,
    ReLUConvNormBlock,
)
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer, get_norm_layer

__all__ = ["DiNTS"]


class _IdentityWithRAMCost(nn.Identity):
    def __init__(self):
        super().__init__()
        self.ram_cost = 0


class _ReLUConvNormBlockWithRAMCost(ReLUConvNormBlock):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int):
        super().__init__(in_channel, out_channel, kernel_size, padding)
        self.ram_cost = 1 + out_channel / in_channel * 2


class _P3DReLUConvNormBlockWithRAMCost(P3DReLUConvNormBlock):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int, padding: int, p3dmode: int = 0):
        super().__init__(in_channel, out_channel, kernel_size, padding, p3dmode)
        self.ram_cost = 1 + 1 + out_channel / in_channel * 2


class _FactorizedIncreaseBlockWithRAMCost(FactorizedIncreaseBlock):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__(in_channel, out_channel)

        # devide by 8 to comply with cell output size
        self.ram_cost = 8 * (1 + 1 + out_channel / in_channel * 2) / 8 * in_channel / out_channel


class _FactorizedReduceBlockWithRAMCost(FactorizedReduceBlock):
    """
    Down-sampling the feature by 2 using stride.
    """

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__(in_channel, out_channel)

        # multiply by 8 to comply with cell output size (see net.get_ram_cost_usage)
        self.ram_cost = (1 + out_channel / in_channel / 8 * 3) * 8 * in_channel / out_channel


# Define Operation Set
OPS = {
    "skip_connect": lambda c: _IdentityWithRAMCost(),
    "conv_3x3x3": lambda c: _ReLUConvNormBlockWithRAMCost(c, c, 3, padding=1),
    "conv_3x3x1": lambda c: _P3DReLUConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=0),
    "conv_3x1x3": lambda c: _P3DReLUConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=1),
    "conv_1x3x3": lambda c: _P3DReLUConvNormBlockWithRAMCost(c, c, 3, padding=1, p3dmode=2),
}


# connection operations
ConnOPS = {
    "up": _FactorizedIncreaseBlockWithRAMCost,
    "down": _FactorizedReduceBlockWithRAMCost,
    "identity": _IdentityWithRAMCost,
    "align_channels": _ReLUConvNormBlockWithRAMCost,
}


class MixedOp(nn.Module):
    def __init__(self, c, arch_arch_code_c=None):
        super().__init__()
        self._ops = nn.ModuleList()
        if arch_arch_code_c is None:
            arch_arch_code_c = np.ones(len(OPS))
        for idx, _ in enumerate(OPS.keys()):
            if idx < len(arch_arch_code_c):
                if arch_arch_code_c[idx] == 0:
                    op = None
                else:
                    op = OPS[_](c)
                self._ops.append(op)

    def forward(self, x, ops=None, weight: float = None):
        pos = (ops == 1).nonzero()
        result = 0
        for _ in pos:
            result += self._ops[_.item()](x) * ops[_.item()] * weight[_.item()]
        return result


class Cell(nn.Module):
    """
    The basic class for cell operation
    Args:
        c_prev: input channel number
        C: output channel number
        rate: resolution change rate. -1 for 2x downsample, 1 for 2x upsample
              0 for no change of resolution
        arch_code_c: cell operation code
    """

    def __init__(self, c_prev, c, rate: int, arch_code_c: bool = None):
        super().__init__()
        self.c_out = c
        if rate == -1:  # downsample
            self.preprocess = ConnOPS["down"](c_prev, c)
        elif rate == 1:  # upsample
            self.preprocess = ConnOPS["up"](c_prev, c)
        else:
            if c_prev == c:
                self.preprocess = ConnOPS["identity"]()
            else:
                self.preprocess = ConnOPS["align_channels"](c_prev, c, 1, 0)
        self.op = MixedOp(c, arch_code_c)

    def forward(self, s, ops, weight):
        s = self.preprocess(s)
        s = self.op(s, ops, weight)
        return s

class Stem(nn.Module):
    """
    The pre-defined stem module for: 1) input downsample 2) output upsample to original size
    Args:
        in_channels: input image channel
        num_classes: output segmentation channel
        filter_nums: filter numbers on each depth (spatial resolution)
        num_depths: number of depths
        spatial_dims: spatial dimenstion of data (3D or 2D)
        use_downsample: use a 2x downsample before dints search space to reduce search GPU ram
        kernel_size: convolution kernel size
        norm_name: normalization name
        padding: convolution padding size
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 filter_nums: list,
                 num_depths: int,
                 spatial_dims: int,
                 act_name: Union[Tuple, str] = "RELU",
                 use_downsample: bool =True,
                 kernel_size:int = 3,
                 norm_name: Union[Tuple, str] = "INSTANCE",
                 padding:int = 1):
        super().__init__()
        # define stem operations for every block
        conv_type = Conv[Conv.CONV, spatial_dims]
        self.stem_down = nn.ModuleDict()
        self.stem_up = nn.ModuleDict()
        self.stem_finals = nn.Sequential(
            get_act_layer(name=act_name),
            conv_type(
                in_channels=filter_nums[0],
                out_channels=filter_nums[0],
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1,
                bias=False,
                dilation=1,
            ),
            get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=filter_nums[0]),
            conv_type(
                in_channels=filter_nums[0],
                out_channels=num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                bias=True,
                dilation=1,
            ),
        )
        for res_idx in range(num_depths):
            # define downsample stems before Dints serach
            if use_downsample:
                self.stem_down[str(res_idx)] = nn.Sequential(
                    nn.Upsample(scale_factor=1 / (2 ** res_idx), mode="trilinear", align_corners=True),
                    conv_type(
                        in_channels=in_channels,
                        out_channels=filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=filter_nums[res_idx]),
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=filter_nums[res_idx],
                        out_channels=filter_nums[res_idx + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=filter_nums[res_idx + 1]),
                )
                self.stem_up[str(res_idx)] = nn.Sequential(
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=filter_nums[res_idx + 1],
                        out_channels=filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=filter_nums[res_idx]),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
                )

            else:
                self.stem_down[str(res_idx)] = nn.Sequential(
                    nn.Upsample(scale_factor=1 / (2 ** res_idx), mode="trilinear", align_corners=True),
                    conv_type(
                        in_channels=in_channels,
                        out_channels=filter_nums[res_idx],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=filter_nums[res_idx]),
                )
                self.stem_up[str(res_idx)] = nn.Sequential(
                    get_act_layer(name=act_name),
                    conv_type(
                        in_channels=filter_nums[res_idx],
                        out_channels=filter_nums[max(res_idx - 1, 0)],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=1,
                        bias=False,
                        dilation=1,
                    ),
                    get_norm_layer(name=norm_name, spatial_dims=spatial_dims, channels=filter_nums[res_idx - 1]),
                    nn.Upsample(scale_factor=2 ** (res_idx != 0), mode="trilinear", align_corners=True),
                )
    def forward():
        pass

class DiNTS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        act_name: Union[Tuple, str] = "RELU",
        channel_mul: float = 1.0,
        cell=Cell,
        cell_ops: int = 5,
        arch_code: list = None,
        norm_name: Union[Tuple, str] = "INSTANCE",
        num_blocks: int = 6,
        num_depths: int = 3,
        spatial_dims: int = 3,
        use_downsample: bool = False,
    ):
        """
        Initialize NAS network search space

        Args:
            in_channels: input image channel
            num_classes: number of segmentation classes
            num_blocks: number of blocks (depth in the horizontal direction)
            num_depths: number of image resolutions: 1, 1/2, 1/4 ... in each dimention, each resolution feature
                is a node at each block
            cell: operatoin of each node
            cell_ops: cell operation numbers
            channel_mul: adjust intermediate channel number, default 1.
            arch_code: [node_a, arch_code_a, arch_code_c] decoded using self.decode(). Remove unused cells in retraining

        Predefined variables:
            filter_nums: default init 64. Double channel number after downsample
            topology related varaibles from gen_mtx():
                trans_mtx: feasible path activation given node activation key
                arch_code2in: path activation to its incoming node index
                arch_code2ops: path activation to operations of upsample 1, keep 0, downsample -1
                arch_code2out: path activation to its output node index
                node_act_list: all node activation arch_codes [2^num_depths-1, res_num]
                node_act_dict: node activation arch_code to its index
                tidx: index used to convert path activation matrix (depth,depth) in trans_mtx to path activation
                    arch_code (1,3*depth-2)
        """
        super().__init__()

        # if searching architecture
        self.is_search = True

        # predefined variables
        filter_nums = [
            int(32 * channel_mul),
            int(64 * channel_mul),
            int(128 * channel_mul),
            int(256 * channel_mul),
            int(512 * channel_mul),
        ]

        # path activation and node activations
        trans_mtx, node_act_list, tidx, arch_code2in, arch_code2ops, arch_code2out, child_list = self.gen_mtx(
            num_depths
        )
        node_act_list = np.array(node_act_list)
        node_act_dict = {str(node_act_list[i]): i for i in range(len(node_act_list))}

        self.num_depths = num_depths
        self.filter_nums = filter_nums
        self.cell_ops = cell_ops
        self.arch_code2in = arch_code2in
        self.arch_code2ops = arch_code2ops
        self.arch_code2out = arch_code2out
        self.node_act_list = node_act_list
        self.node_act_dict = node_act_dict
        self.trans_mtx = trans_mtx
        self.tidx = tidx
        self.child_list = np.array(child_list)
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.num_depths = num_depths
        self.use_downsample = use_downsample

        # define stem operations for every block
        self.stem = Stem(in_channels=in_channels, num_classes=num_classes, filter_nums=filter_nums, num_depths=num_depths,
                         spatial_dims=spatial_dims, act_name=act_name, use_downsample=use_downsample, norm_name=norm_name)

        # define NAS search space
        if arch_code is None:
            arch_code_a = np.ones((num_blocks, len(arch_code2out)))
            arch_code_c = np.ones((num_blocks, len(arch_code2out), cell_ops))
        else:
            arch_code_a = arch_code[1]
            arch_code_c = F.one_hot(torch.from_numpy(arch_code[2]), cell_ops).numpy()
        self.cell_tree = nn.ModuleDict()
        self.ram_cost = np.zeros((num_blocks, len(arch_code2out), cell_ops))
        for blk_idx in range(num_blocks):
            for res_idx in range(len(arch_code2out)):
                if arch_code_a[blk_idx, res_idx] == 1:
                    self.cell_tree[str((blk_idx, res_idx))] = cell(
                        filter_nums[arch_code2in[res_idx] + int(use_downsample)],
                        filter_nums[arch_code2out[res_idx] + int(use_downsample)],
                        arch_code2ops[res_idx],
                        arch_code_c[blk_idx, res_idx],
                    )
                    self.ram_cost[blk_idx, res_idx] = np.array(
                        [
                            _.ram_cost + self.cell_tree[str((blk_idx, res_idx))].preprocess.ram_cost
                            if _ is not None
                            else 0
                            for _ in self.cell_tree[str((blk_idx, res_idx))].op._ops[:cell_ops]
                        ]
                    )

        # define cell and macro arhitecture probabilities
        self.log_alpha_c = nn.Parameter(
            torch.zeros(num_blocks, len(arch_code2out), cell_ops).normal_(1, 0.01).cuda().requires_grad_()
        )
        self.log_alpha_a = nn.Parameter(
            torch.zeros(num_blocks, len(arch_code2out)).normal_(0, 0.01).cuda().requires_grad_()
        )
        self._arch_param_names = ["log_alpha_a", "log_alpha_c"]

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if name not in self._arch_param_names]

    def get_prob_a(self, child: bool = False):
        """
        Get final path probabilities and child model weights

        Args:
            child: return child probability as well (used in decode)
        """
        log_alpha = self.log_alpha_a
        _arch_code_prob_a = torch.sigmoid(log_alpha)
        norm = 1 - (1 - _arch_code_prob_a).prod(-1)  # normalizing factor
        arch_code_prob_a = _arch_code_prob_a / norm.unsqueeze(1)
        if child:
            probs_a = []
            path_activation = torch.from_numpy(self.child_list).cuda()
            for blk_idx in range(self.num_blocks):
                probs_a.append(
                    (
                        path_activation * _arch_code_prob_a[blk_idx]
                        + (1 - path_activation) * (1 - _arch_code_prob_a[blk_idx])
                    ).prod(-1)
                    / norm[blk_idx]
                )
            probs_a = torch.stack(probs_a)
            return probs_a, arch_code_prob_a
        else:
            return None, arch_code_prob_a

    def get_ram_cost_usage(self, in_size, cell_ram_cost: bool = False, arch_code: bool = None, full: bool = False):
        """
        Get estimated output tensor size

        Args:
            in_size: input image shape at the highest resolutoin level
            full: full ram cost usage with all probability of 1
        """
        # convert input image size to feature map size at each level
        b, c, h, w, s = in_size
        sizes = []
        for res_idx in range(self.num_depths):
            sizes.append(
                b * self.filter_nums[res_idx] * h // (2 ** res_idx) * w // (2 ** res_idx) * s // (2 ** res_idx)
            )
        sizes = torch.tensor(sizes).to(torch.float32).cuda() // (2 ** (int(self.use_downsample)))
        probs_a, arch_code_prob_a = self.get_prob_a(child=False)
        cell_prob = F.softmax(self.log_alpha_c, dim=-1)
        if full:
            arch_code_prob_a = arch_code_prob_a.detach()
            arch_code_prob_a.fill_(1)
            if cell_ram_cost:
                cell_prob = cell_prob.detach()
                cell_prob.fill_(1 / self.cell_ops)
        ram_cost = torch.from_numpy(self.ram_cost).to(torch.float32).cuda()
        usage = 0
        for blk_idx in range(self.num_blocks):
            # node activation for input
            # cell operation
            for path_idx in range(len(self.arch_code2out)):
                if arch_code is not None:
                    usage += (
                        arch_code[0][blk_idx, path_idx]
                        * (1 + (ram_cost[blk_idx, path_idx] * arch_code[1][blk_idx, path_idx]).sum())
                        * sizes[self.arch_code2out[path_idx]]
                    )
                else:
                    usage += (
                        arch_code_prob_a[blk_idx, path_idx]
                        * (1 + (ram_cost[blk_idx, path_idx] * cell_prob[blk_idx, path_idx]).sum())
                        * sizes[self.arch_code2out[path_idx]]
                    )
        return usage * 32 / 8 / 1024 ** 2

    def get_topology_entropy(self, probs: float):
        """
        Get topology entropy loss
        """
        if hasattr(self, "node2in"):
            node2in = self.node2in
            node2out = self.node2out
        else:
            # node activation index to feasible input child_idx
            node2in = [[] for i in range(len(self.node_act_list))]
            # node activation index to feasible output child_idx
            node2out = [[] for i in range(len(self.node_act_list))]
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
        Decode network log_alpha_a/log_alpha_c using dijkstra shortpath algorithm

        Return:
            arch_code with maximum probability
        """
        probs, arch_code_prob_a = self.get_prob_a(child=True)
        arch_code_a_max = self.child_list[torch.argmax(probs, -1).data.cpu().numpy()]
        arch_code_c = torch.argmax(F.softmax(self.log_alpha_c, -1), -1).data.cpu().numpy()
        probs = probs.data.cpu().numpy()

        # define adacency matrix
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
            for mtx in self.trans_mtx[str(_node_act)]:
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

        # solving shortest path problem
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import dijkstra

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

    def gen_mtx(self, depth: int = 3):
        """
        Generate elements needed in decoding, topology e.t.c
        """
        # total path in a block
        paths = 3 * depth - 2

        # use depth first search to find all path activation combination
        def dfs(node, paths=6):
            if node == paths:
                return [[0], [1]]
            else:
                child = dfs(node + 1, paths)
                return [[0] + _ for _ in child] + [[1] + _ for _ in child]

        all_connect = dfs(0, paths - 1)

        # Save all possible connections in mtx (might be redundant and infeasible)
        mtx = []
        for _ in all_connect:
            # convert path activation [1,paths] to path activation matrix [depth, depth]
            ma = np.zeros((depth, depth))
            for i in range(paths):
                ma[(i + 1) // 3, (i + 1) // 3 - 1 + (i + 1) % 3] = _[i]
            mtx.append(ma)

        # Calculate path activation to node activation params
        tidx, arch_code2in, arch_code2out = [], [], []
        for i in range(paths):
            tidx.append((i + 1) // 3 * depth + (i + 1) // 3 - 1 + (i + 1) % 3)
            arch_code2in.append((i + 1) // 3 - 1 + (i + 1) % 3)
        arch_code2ops = ([-1, 0, 1] * depth)[1:-1]
        for _ in range(depth):
            arch_code2out.extend([_, _, _])
        arch_code2out = arch_code2out[1:-1]

        # define all possible node activativation
        node_act_list = dfs(0, depth - 1)[1:]
        transfer_mtx = {}
        for arch_code in node_act_list:
            # make sure each activated node has an active connection, inactivated node has no connection
            arch_code_mtx = [_ for _ in mtx if ((np.sum(_, 0) > 0).astype(int) == np.array(arch_code)).all()]
            transfer_mtx[str(np.array(arch_code))] = arch_code_mtx

        return transfer_mtx, node_act_list, tidx, arch_code2in, arch_code2ops, arch_code2out, all_connect[1:]

    def forward(self, x, arch_code: list = None):
        """
        Prediction based on dynamic arch_code

        Args:
            x: input tensor
            arch_code: [node_a, arch_code_a, arch_code_c]
        """
        # define output positions
        out_pos = [self.num_blocks - 1]

        # sample path weights
        predict_all = []
        node_a, arch_code_a, arch_code_c = arch_code
        probs_a, arch_code_prob_a = self.get_prob_a(child=False)

        # stem inference
        inputs = []
        for _ in range(self.num_depths):
            # allow multi-resolution input
            if node_a[0][_]:
                inputs.append(self.stem.stem_down[str(_)](x))
            else:
                inputs.append(None)

        for blk_idx in range(self.num_blocks):
            outputs = [0] * self.num_depths
            for res_idx, activation in enumerate(arch_code_a[blk_idx].data.cpu().numpy()):
                if activation:
                    if self.is_search:
                        outputs[self.arch_code2out[res_idx]] += (
                            self.cell_tree[str((blk_idx, res_idx))](
                                inputs[self.arch_code2in[res_idx]],
                                ops=arch_code_c[blk_idx, res_idx],
                                weight=F.softmax(self.log_alpha_c[blk_idx, res_idx], dim=-1),
                            )
                            * arch_code_prob_a[blk_idx, res_idx]
                        )
                    else:
                        outputs[self.arch_code2out[res_idx]] += self.cell_tree[str((blk_idx, res_idx))](
                            inputs[self.arch_code2in[res_idx]],
                            ops=arch_code_c[blk_idx, res_idx],
                            weight=torch.ones_like(arch_code_c[blk_idx, res_idx], requires_grad=False),
                        )
            inputs = outputs
            if blk_idx in out_pos:
                start = False
                for res_idx in range(self.num_depths - 1, -1, -1):
                    if start:
                        _temp = self.stem.stem_up[str(res_idx)](inputs[res_idx] + _temp)
                    elif node_a[blk_idx + 1][res_idx]:
                        start = True
                        _temp = self.stem.stem_up[str(res_idx)](inputs[res_idx])
                prediction = self.stem.stem_finals(_temp)
                predict_all.append(prediction)
        return predict_all
