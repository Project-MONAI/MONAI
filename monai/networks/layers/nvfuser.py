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
from torch import Tensor
from torch.nn.modules.batchnorm import _NormBase

from monai.utils import optional_import

instance_norm_nvfuser_cuda, _ = optional_import("instance_norm_nvfuser_cuda")

__all__ = ["InstanceNorm3dNVFuser"]


class InstanceNormNVFuserFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        use_input_stats: bool,
        momentum: float,
        eps: float,
    ):

        channels_last = input.is_contiguous(memory_format=torch.channels_last) or input.is_contiguous(
            memory_format=torch.channels_last_3d
        )
        # for channels_last format input, reorder it into NCHW[D] format
        if channels_last:
            order = [0] + [i for i in range(2, len(input.shape))] + [1]
            _input = input.permute(order)
        else:
            _input = input
        if not _input.is_contiguous():
            raise AssertionError("In NCHW[D] order, `input` must be contiguous.")
        result = instance_norm_nvfuser_cuda.forward(
            _input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, channels_last
        )
        if len(result) == 3:
            out, mean, invstd = result
        else:
            running_mean, running_var, out, mean, invstd = result
        ctx.use_input_stats = use_input_stats
        ctx.eps = eps
        ctx.channels_last = channels_last
        # saving for backward in "explicit channels-last format"
        ctx.save_for_backward(_input, weight, running_mean, running_var, mean, invstd)
        if channels_last:
            order = [0, len(_input.shape) - 1] + [i for i in range(1, len(_input.shape) - 1)]
            out = out.permute(order)

            if len(out.shape) == 4:
                memory_format = torch.channels_last
            elif len(out.shape) == 5:
                memory_format = torch.channels_last_3d
            else:
                raise AssertionError("unhandled channels_last format variation in forward.")
            if not out.is_contiguous(memory_format=memory_format):
                raise AssertionError(f"In {memory_format} order, output of forward is not contiguous.")
            if not input.is_contiguous(memory_format=memory_format):
                raise AssertionError(f"In {memory_format} order, `input` is not contiguous.")

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore

        if ctx.channels_last:
            order = [0] + [i for i in range(2, len(grad_output.shape))] + [1]
            grad_output = grad_output.permute(order)
        # input was saved in "explicit channels-last format"
        if not ctx.saved_tensors[0].is_contiguous():
            raise AssertionError("In NCHW order, `ctx.saved_tensors[0]` is not contiguous.")
        grad_output = grad_output.contiguous()
        saved = list(ctx.saved_tensors)
        saved.insert(1, grad_output)
        grad_input, grad_weight, grad_bias = instance_norm_nvfuser_cuda.backward(
            *saved, ctx.use_input_stats, ctx.eps, ctx.channels_last
        )
        if ctx.channels_last:
            order = [0, len(grad_input.shape) - 1] + [i for i in range(1, len(grad_input.shape) - 1)]
            grad_input = grad_input.permute(order)
            if len(grad_input.shape) == 4:
                memory_format = torch.channels_last
            elif len(grad_input.shape) == 5:
                memory_format = torch.channels_last_3d
            else:
                raise AssertionError("unhandled channels_last format variation in backward.")
            if not grad_input.is_contiguous(memory_format=memory_format):
                raise AssertionError(f"In {memory_format} order, output of backward is not contiguous.")

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class _InstanceNormNVFuser(_NormBase):
    """
    Base of InstanceNorm3dNVFuser. This class only works on non-Windows OS and input tensors should be in GPU mode.
    This class refers to `APEX`.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.dummy = torch.empty([], device="cuda")

    def _check_input_dim(self, input: torch.Tensor):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        # at version 1: removed running_mean and running_var when
        # track_running_stats=False (default)
        if version is None and not self.track_running_stats:
            running_stats_keys = []
            for name in ("running_mean", "running_var"):
                key = prefix + name
                if key in state_dict:
                    running_stats_keys.append(key)
            if len(running_stats_keys) > 0:
                error_msgs.append(
                    "Unexpected running stats buffer(s) {names} for {klass} "
                    "with track_running_stats=False. If state_dict is a "
                    "checkpoint saved before 0.4.0, this may be expected "
                    "because {klass} does not track running stats by default "
                    "since 0.4.0. Please remove these keys from state_dict. If "
                    "the running stats are actually needed, instead set "
                    "track_running_stats=True in {klass} to enable them. See "
                    "the documentation of {klass} for details.".format(
                        names=" and ".join(f"{k}" for k in running_stats_keys), klass=self.__class__.__name__
                    )
                )
                for key in running_stats_keys:
                    state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, input: Tensor):
        if not input.is_cuda:
            raise AssertionError("NVFuser InstanceNorm is CUDA only.")
        self._check_input_dim(input)

        out = InstanceNormNVFuserFunction.apply(
            input,
            self.weight if self.weight is not None else self.dummy,
            self.bias if self.bias is not None else self.dummy,
            self.running_mean if self.running_mean is not None else self.dummy,
            self.running_var if self.running_mean is not None else self.dummy,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )

        return out


class InstanceNorm3dNVFuser(_InstanceNormNVFuser):
    """
    A faster version of 3d instance norm layer.
    This class only works on non-Windows OS and input tensors should be in GPU mode.
    This class refers to `APEX`.
    """

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() != 5:
            raise ValueError(f"expected 5D input (got {input.dim()}D input)")
