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

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dints_block import (
    FactorizedIncreaseBlock,
    FactorizedReduceBlock,
    P3DReLUConvBNBlock,
    ReLUConvBNBlock,
)

__all__ = ["DiNTS"]


class _IdentityWithMemory(nn.Identity):
    def __init__(self):
        super().__init__()
        self.memory = 0


class _ReLUConvBNBlockWithMemory(ReLUConvBNBlock):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, padding: int):
        super().__init__(C_in, C_out, kernel_size, padding)
        self.memory = 1 + C_out / C_in * 2


class _P3DReLUConvBNBlockWithMemory(P3DReLUConvBNBlock):
    def __init__(self, C_in: int, C_out: int, kernel_size: int, padding: int, P3Dmode: int = 0):
        super().__init__(C_in, C_out, kernel_size, padding, P3Dmode)
        self.memory = 1 + 1 + C_out / C_in * 2


class _FactorizedIncreaseBlockWithMemory(FactorizedIncreaseBlock):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__(in_channel, out_channel)

        # devide by 8 to comply with cell output size
        self.memory = 8 * (1 + 1 + out_channel / in_channel * 2) / 8 * in_channel / out_channel


class _FactorizedReduceBlockWithMemory(FactorizedReduceBlock):
    """
    Down-sampling the feature by 2 using stride.
    """

    def __init__(self, C_in: int, C_out: int):
        super().__init__(C_in, C_out)

        # multiply by 8 to comply with cell output size (see net.get_memory_usage)
        self.memory = (1 + C_out / C_in / 8 * 3) * 8 * C_in / C_out


# Define Operation Set
OPS = {
    "skip_connect": lambda C: _IdentityWithMemory(),
    "conv_3x3x3": lambda C: _ReLUConvBNBlockWithMemory(C, C, 3, padding=1),
    "conv_3x3x1": lambda C: _P3DReLUConvBNBlockWithMemory(C, C, 3, padding=1, P3Dmode=0),
    "conv_3x1x3": lambda C: _P3DReLUConvBNBlockWithMemory(C, C, 3, padding=1, P3Dmode=1),
    "conv_1x3x3": lambda C: _P3DReLUConvBNBlockWithMemory(C, C, 3, padding=1, P3Dmode=2),
}

# Define Operation Set
# OPS_2D = {
#     "skip_connect": lambda C: Identity(),
#     "conv_3x3"  : lambda C: ReLUConvBN(C, C, 3, padding=1),
# }


class MixedOp(nn.Module):
    def __init__(self, C, code_c=None):
        super().__init__()
        self._ops = nn.ModuleList()
        if code_c is None:
            code_c = np.ones(len(OPS))
        for idx, _ in enumerate(OPS.keys()):
            if idx < len(code_c):
                if code_c[idx] == 0:
                    op = None
                else:
                    op = OPS[_](C)
                self._ops.append(op)

    def forward(self, x, ops=None, weight: bool = None):
        pos = (ops == 1).nonzero()
        result = 0
        for _ in pos:
            result += self._ops[_.item()](x) * ops[_.item()] * weight[_.item()]
        return result


class Cell(nn.Module):
    """
    The basic class for cell operation
    Args:
        C_prev: input channel number
        C: output channel number
        rate: resolution change rate. -1 for 2x downsample, 1 for 2x upsample
              0 for no change of resolution
        code_c: cell operation code
    """

    def __init__(self, C_prev, C, rate: int, code_c: bool = None):
        super().__init__()
        self.C_out = C
        if rate == -1:  # downsample
            self.preprocess = _FactorizedReduceBlockWithMemory(C_prev, C)
        elif rate == 1:  # upsample
            self.preprocess = _FactorizedIncreaseBlockWithMemory(C_prev, C)
        else:
            if C_prev == C:
                self.preprocess = _IdentityWithMemory()
            else:
                self.preprocess = _ReLUConvBNBlockWithMemory(C_prev, C, 1, 0)
        self.op = MixedOp(C, code_c)

    def forward(self, s, ops, weight):
        s = self.preprocess(s)
        s = self.op(s, ops, weight)
        return s


class DiNTS(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        cell=Cell,
        cell_ops: int = 5,
        channel_mul: float = 1.0,
        num_blocks: int = 6,
        num_depths: int = 3,
        use_stem: bool = False,
        code: list = None,
    ):
        """
        Initialize NAS network search space

        Args:
            in_channels: input image channel
            num_classes: number of segmentation classes
            num_blocks: number of blocks (depth in the horizontal direction)
            num_depths: number of image resolutions: 1, 1/2, 1/4 ... in each dimention, each resolution feature is a node at each block
            cell: operatoin of each node
            cell_ops: cell operation numbers
            channel_mul: adjust intermediate channel number, default 1.
            code: [node_a, code_a, code_c] decoded using self.decode(). Remove unused cells in retraining

        Predefined variables:
            filter_nums: default init 64. Double channel number after downsample
            topology related varaibles from gen_mtx():
                trans_mtx: feasible path activation given node activation key
                code2in: path activation to its incoming node index
                code2ops: path activation to operations of upsample 1, keep 0, downsample -1
                code2out: path activation to its output node index
                node_act_list: all node activation codes [2^num_depths-1, res_num]
                node_act_dict: node activation code to its index
                tidx: index used to convert path activation matrix (depth,depth) in trans_mtx to path activation code (1,3*depth-2)
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
        trans_mtx, node_act_list, tidx, code2in, code2ops, code2out, child_list = self._gen_mtx(num_depths)
        node_act_list = np.array(node_act_list)
        node_act_dict = {str(node_act_list[i]): i for i in range(len(node_act_list))}

        self.num_depths = num_depths
        self.filter_nums = filter_nums
        self.cell_ops = cell_ops
        self.code2in = code2in
        self.code2ops = code2ops
        self.code2out = code2out
        self.node_act_list = node_act_list
        self.node_act_dict = node_act_dict
        self.trans_mtx = trans_mtx
        self.tidx = tidx
        self.child_list = np.array(child_list)
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.num_depths = num_depths
        self.use_stem = use_stem

        # define stem operations for every block
        self.stem_down = nn.ModuleDict()
        self.stem_up = nn.ModuleDict()
        self.stem_finals = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(filter_nums[0], filter_nums[0], 3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(filter_nums[0]),
            nn.Conv3d(filter_nums[0], num_classes, 1, stride=1, padding=0, bias=True),
        )
        for res_idx in range(num_depths):
            if use_stem:
                self.stem_down[str(res_idx)] = nn.Sequential(
                    nn.Upsample(scale_factor=1 / (2 ** res_idx), mode="trilinear", align_corners=True),
                    nn.Conv3d(in_channels, filter_nums[res_idx], 3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm3d(filter_nums[res_idx]),
                    nn.ReLU(),
                    nn.Conv3d(filter_nums[res_idx], filter_nums[res_idx + 1], 3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(filter_nums[res_idx + 1]),
                )
                self.stem_up[str(res_idx)] = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv3d(filter_nums[res_idx + 1], filter_nums[res_idx], 3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm3d(filter_nums[res_idx]),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
                )

            else:
                self.stem_down[str(res_idx)] = nn.Sequential(
                    nn.Upsample(scale_factor=1 / (2 ** res_idx), mode="trilinear", align_corners=True),
                    nn.Conv3d(in_channels, filter_nums[res_idx], 3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm3d(filter_nums[res_idx]),
                )
                self.stem_up[str(res_idx)] = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv3d(filter_nums[res_idx], filter_nums[res_idx], 3, stride=1, padding=1, bias=False),
                    nn.InstanceNorm3d(filter_nums[res_idx]),
                    nn.Conv3d(filter_nums[res_idx], num_classes, 1),
                    nn.Upsample(scale_factor=2 ** res_idx, mode="trilinear", align_corners=True),
                )

        # define NAS search space
        if code is None:
            code_a = np.ones((num_blocks, len(code2out)))
            code_c = np.ones((num_blocks, len(code2out), cell_ops))
        else:
            code_a = code[1]
            code_c = F.one_hot(torch.from_numpy(code[2]), cell_ops).numpy()
        self.cell_tree = nn.ModuleDict()
        self.memory = np.zeros((num_blocks, len(code2out), cell_ops))
        for blk_idx in range(num_blocks):
            for res_idx in range(len(code2out)):
                if code_a[blk_idx, res_idx] == 1:
                    self.cell_tree[str((blk_idx, res_idx))] = cell(
                        filter_nums[code2in[res_idx] + int(use_stem)],
                        filter_nums[code2out[res_idx] + int(use_stem)],
                        code2ops[res_idx],
                        code_c[blk_idx, res_idx],
                    )
                    self.memory[blk_idx, res_idx] = np.array(
                        [
                            _.memory + self.cell_tree[str((blk_idx, res_idx))].preprocess.memory if _ is not None else 0
                            for _ in self.cell_tree[str((blk_idx, res_idx))].op._ops[:cell_ops]
                        ]
                    )

        # define cell and macro arhitecture probabilities
        self.log_alpha_c = torch.nn.Parameter(
            torch.zeros(num_blocks, len(code2out), cell_ops).normal_(1, 0.01).cuda().requires_grad_()
        )
        self.log_alpha_a = torch.nn.Parameter(
            torch.zeros(num_blocks, len(code2out)).normal_(0, 0.01).cuda().requires_grad_()
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
        _code_prob_a = torch.sigmoid(log_alpha)
        norm = 1 - (1 - _code_prob_a).prod(-1)  # normalizing factor
        code_prob_a = _code_prob_a / norm.unsqueeze(1)
        if child:
            probs_a = []
            path_activation = torch.from_numpy(self.child_list).cuda()
            for blk_idx in range(self.num_blocks):
                probs_a.append(
                    (
                        path_activation * _code_prob_a[blk_idx] + (1 - path_activation) * (1 - _code_prob_a[blk_idx])
                    ).prod(-1)
                    / norm[blk_idx]
                )
            probs_a = torch.stack(probs_a)
            return probs_a, code_prob_a
        else:
            return None, code_prob_a

    def _get_memory_usage(self, in_size, cell_memory: bool = False, code: bool = None, full: bool = False):
        """
        Get estimated output tensor size

        Args:
            in_size: input image shape at the highest resolutoin level
            full: full memory usage with all probability of 1
        """
        # convert input image size to feature map size at each level
        b, c, h, w, s = in_size
        sizes = []
        for res_idx in range(self.num_depths):
            sizes.append(
                b * self.filter_nums[res_idx] * h // (2 ** res_idx) * w // (2 ** res_idx) * s // (2 ** res_idx)
            )
        sizes = torch.tensor(sizes).to(torch.float32).cuda() // (2 ** (int(self.use_stem)))
        probs_a, code_prob_a = self.get_prob_a(child=False)
        cell_prob = F.softmax(self.log_alpha_c, dim=-1)
        if full:
            code_prob_a = code_prob_a.detach()
            code_prob_a.fill_(1)
            if cell_memory:
                cell_prob = cell_prob.detach()
                cell_prob.fill_(1 / self.cell_ops)
        memory = torch.from_numpy(self.memory).to(torch.float32).cuda()
        usage = 0
        for blk_idx in range(self.num_blocks):
            # node activation for input
            # cell operation
            for path_idx in range(len(self.code2out)):
                if code is not None:
                    usage += (
                        code[0][blk_idx, path_idx]
                        * (1 + (memory[blk_idx, path_idx] * code[1][blk_idx, path_idx]).sum())
                        * sizes[self.code2out[path_idx]]
                    )
                else:
                    usage += (
                        code_prob_a[blk_idx, path_idx]
                        * (1 + (memory[blk_idx, path_idx] * cell_prob[blk_idx, path_idx]).sum())
                        * sizes[self.code2out[path_idx]]
                    )
        return usage * 32 / 8 / 1024 ** 2

    def _get_topology_entropy(self, probs: float):
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
                for res_idx in range(len(self.code2out)):
                    _node_out[self.code2out[res_idx]] += self.child_list[child_idx][res_idx]
                    _node_in[self.code2in[res_idx]] += self.child_list[child_idx][res_idx]
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

    def _decode(self):
        """
        Decode network log_alpha_a/log_alpha_c using dijkstra shortpath algorithm

        Return:
            code with maximum probability
        """
        probs, code_prob_a = self.get_prob_a(child=True)
        code_a_max = self.child_list[torch.argmax(probs, -1).data.cpu().numpy()]
        code_c = torch.argmax(F.softmax(self.log_alpha_c, -1), -1).data.cpu().numpy()
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
                _node_act[self.code2out[path_idx]] += self.child_list[child_idx][path_idx]
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
        code_a = np.zeros((self.num_blocks, len(self.code2out)))
        node_a = np.zeros((self.num_blocks + 1, self.num_depths))

        # decoding to paths
        while True:
            index = predecessors[index]
            if index == 0:
                break
            child_idx = (index - 1) % len(self.child_list)
            code_a[a_idx, :] = self.child_list[child_idx]
            for res_idx in range(len(self.code2out)):
                node_a[a_idx, self.code2out[res_idx]] += code_a[a_idx, res_idx]
            a_idx -= 1
        for res_idx in range(len(self.code2out)):
            node_a[a_idx, self.code2in[res_idx]] += code_a[0, res_idx]
        node_a = (node_a >= 1).astype(int)
        return node_a, code_a, code_c, code_a_max

    def _gen_mtx(self, depth: int = 3):
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
        tidx, code2in, code2out = [], [], []
        for i in range(paths):
            tidx.append((i + 1) // 3 * depth + (i + 1) // 3 - 1 + (i + 1) % 3)
            code2in.append((i + 1) // 3 - 1 + (i + 1) % 3)
        code2ops = ([-1, 0, 1] * depth)[1:-1]
        for _ in range(depth):
            code2out.extend([_, _, _])
        code2out = code2out[1:-1]

        # define all possible node activativation
        node_act_list = dfs(0, depth - 1)[1:]
        transfer_mtx = {}
        for code in node_act_list:
            # make sure each activated node has an active connection, inactivated node has no connection
            code_mtx = [_ for _ in mtx if ((np.sum(_, 0) > 0).astype(int) == np.array(code)).all()]
            transfer_mtx[str(np.array(code))] = code_mtx

        return transfer_mtx, node_act_list, tidx, code2in, code2ops, code2out, all_connect[1:]

    def forward(self, x, code: list = None):
        """
        Prediction based on dynamic code

        Args:
            x: input tensor
            code: [node_a, code_a, code_c]
        """
        # define output positions
        out_pos = [self.num_blocks - 1]

        # sample path weights
        predict_all = []
        node_a, code_a, code_c = code
        probs_a, code_prob_a = self.get_prob_a(child=False)

        # stem inference
        inputs = []
        for _ in range(self.num_depths):
            # allow multi-resolution input
            if node_a[0][_]:
                inputs.append(self.stem_down[str(_)](x))
            else:
                inputs.append(None)

        for blk_idx in range(self.num_blocks):
            outputs = [0] * self.num_depths
            for res_idx, activation in enumerate(code_a[blk_idx].data.cpu().numpy()):
                if activation:
                    if self.is_search:
                        outputs[self.code2out[res_idx]] += (
                            self.cell_tree[str((blk_idx, res_idx))](
                                inputs[self.code2in[res_idx]],
                                ops=code_c[blk_idx, res_idx],
                                weight=F.softmax(self.log_alpha_c[blk_idx, res_idx], dim=-1),
                            )
                            * code_prob_a[blk_idx, res_idx]
                        )
                    else:
                        outputs[self.code2out[res_idx]] += self.cell_tree[str((blk_idx, res_idx))](
                            inputs[self.code2in[res_idx]],
                            ops=code_c[blk_idx, res_idx],
                            weight=torch.ones_like(code_c[blk_idx, res_idx], requires_grad=False),
                        )
            inputs = outputs
            if blk_idx in out_pos:
                start = False
                for res_idx in range(self.num_depths - 1, -1, -1):
                    if start:
                        _temp = self.stem_up[str(res_idx)](inputs[res_idx] + _temp)
                    elif node_a[blk_idx + 1][res_idx]:
                        start = True
                        _temp = self.stem_up[str(res_idx)](inputs[res_idx])
                prediction = self.stem_finals(_temp)
                predict_all.append(prediction)
        return predict_all
